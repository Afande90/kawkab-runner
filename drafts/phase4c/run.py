# Phase 4c = Phase 4b revised per AMENDMENT 1 (k1-5 training, k4/k5 oversampled, no "next" queries, value-only
# output head, train to criterion, data pool generated once in parallel, subspace-interchange Gate C).
# Phase 4b — pre-registered Step 2 (PREREGISTRATION-phase4.md): train a variable-binding TEACHER from scratch on answers
# only, then run the gates BEFORE any student exists:
#   A  linear probe for the hop-j variable at every layer (last position); pass if >=90% somewhere, best layer rises with j
#   B  probe controls: random-init teacher, shuffled labels -> near chance
#   C  interchange patching: same statements, two different query chains; copy the layer-L last-position state from
#      example B into A; pass if the answer switches to B's answer in >70% of cases at the probe-best layer
#   +  probe of the hop variable from the final logits (should be ~chance: the teacher never outputs the step)
#   +  lens gate: do the 32 Jacobian-lens scores (and logit-lens scores) decode the hop variable?
import json, os, random, time
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

TEST = os.environ.get("KAWKAB_TEST") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
OUT = {"phase": "4c", "students": {}}

NV, NVAL, NSTMT = 24, 8, 12
TRAIN_K, ALL_K = (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7)
K_WEIGHTS = (0.4 / 3, 0.4 / 3, 0.4 / 3, 0.3, 0.3)
P_NEXT = 0.0
LAYERS, WIDTH, HEADS = (8, 256, 8) if not TEST else (3, 32, 4)
STEPS, BATCH, LR = (120000, 512, 1e-3) if not TEST else (60, 32, 1e-3)
POOL = 2_000_000 if not TEST else 4000
EVAL_EVERY = 1000 if not TEST else 30
N_PROBE_FIT, N_PROBE_TEST = (20000, 5000) if not TEST else (300, 100)
N_PATCH = 2000 if not TEST else 50

# vocabulary: vars 0..23, values 24..31, then specials
V = list(range(NV)); D = list(range(NV, NV + NVAL))
EQ, SEMI, Q, COLON, NEXT, PAD = range(NV + NVAL, NV + NVAL + 6)
VOCAB = NV + NVAL + 6


def save():
    json.dump(OUT, open("phase4c.json", "w"), indent=1)


def make_example(k, rng, query_next=False, second_chain=False):
    """Returns tokens (list), answer token, chain (list of k vars from query var to last var), and optionally a
    second chain of the same depth in the same statement set (for patching)."""
    vars_ = rng.sample(V, NV)
    used = 0

    def chain(depth):
        nonlocal used
        vs = vars_[used:used + depth]; used += depth
        val = rng.choice(D)
        st = [(vs[i], vs[i + 1]) for i in range(depth - 1)] + [(vs[-1], val)]
        return vs, val, st

    tv, tval, stmts = chain(k)
    other = None
    if second_chain:
        ov, oval, ost = chain(k)
        stmts += ost; other = (ov, oval)
    while len(stmts) < NSTMT:
        dd = rng.randint(1, max(1, min(k, NSTMT - len(stmts))))
        if used + dd > NV:
            break
        _, _, st = chain(dd)
        stmts += st
    rng.shuffle(stmts)
    toks = []
    for a, b in stmts:
        toks += [a, EQ, b, SEMI]
    if query_next:
        a, b = rng.choice(stmts)
        toks += [Q, NEXT, a, COLON]
        return toks, b, None, None
    toks += [Q, tv[0], COLON]
    return toks, tval, tv, other


def batch_tensor(seqs):
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, n - len(s):] = torch.tensor(s)          # left-pad so the answer position is always the last
    return x


def gen_batch(n, rng, ks=TRAIN_K, p_next=P_NEXT, weights=None):
    seqs, ans = [], []
    for _ in range(n):
        k = rng.choices(ks, weights=weights)[0] if weights else rng.choice(ks)
        if rng.random() < p_next:
            t, a, _, _ = make_example(k, rng, query_next=True)
        else:
            t, a, _, _ = make_example(k, rng)
        seqs.append(t); ans.append(a)
    return batch_tensor(seqs).to(dev), torch.tensor(ans, device=dev)


def _pool_part(args):
    n, seed = args
    rng = random.Random(seed)
    seqs, ans = [], []
    for _ in range(n):
        k = rng.choices(TRAIN_K, weights=K_WEIGHTS)[0]
        t, a, _, _ = make_example(k, rng)
        seqs.append(t); ans.append(a)
    return batch_tensor(seqs), torch.tensor(ans)


def build_pool():
    import multiprocessing as mp
    parts = max(1, os.cpu_count() or 1)
    with mp.Pool(parts) as p:
        res = p.map(_pool_part, [(POOL // (parts * 8), 1000 + i) for i in range(parts * 8)])
    n = max(x.shape[1] for x, _ in res)
    xs = [torch.cat([torch.full((x.shape[0], n - x.shape[1]), PAD, dtype=torch.long), x], 1) for x, _ in res]
    return torch.cat(xs).to(dev), torch.cat([a for _, a in res]).to(dev)


def new_model(seed):
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=VOCAB, hidden_size=WIDTH, num_hidden_layers=LAYERS, num_attention_heads=HEADS,
                        intermediate_size=4 * WIDTH, max_position_embeddings=128, rotary_pct=0.25,
                        use_parallel_residual=True, tie_word_embeddings=False)
    return GPTNeoXForCausalLM(cfg).to(dev)


def logits_last(model, x):
    """Value-only output head: logits over the 8 value tokens (index 0..7 = token NV..NV+7)."""
    return model(input_ids=x, use_cache=False).logits[:, -1, NV:NV + NVAL].float()


@torch.no_grad()
def accuracy(model, k, n, seed, p_next=0.0):
    model.eval()
    rng = random.Random(seed)
    correct = 0
    for i in range(0, n, 500):
        x, a = gen_batch(min(500, n - i), rng, ks=(k,), p_next=p_next)
        correct += (logits_last(model, x).argmax(-1) == a - NV).sum().item()
    return correct / n


def train_teacher():
    model = new_model(0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 500))
    log, t0 = [], time.time()
    px, pa = build_pool()
    print("pool", tuple(px.shape), f"({time.time() - t0:.0f}s)", flush=True)
    gen = torch.Generator(device=dev).manual_seed(1)
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
    for step in range(STEPS):
        model.train()
        idx = torch.randint(0, len(px), (BATCH,), device=dev, generator=gen)
        x, a = px[idx], pa[idx]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            loss = F.cross_entropy(logits_last(model, x), a - NV)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % EVAL_EVERY == 0:
            acc = {k: accuracy(model, k, 2000 if not TEST else 100, 99) for k in TRAIN_K}
            row = {"step": step + 1, "loss": round(loss.item(), 4), "acc": acc, "sec": round(time.time() - t0)}
            log.append(row); print(row, flush=True)
            if not TEST and all(v >= 0.993 for v in acc.values()):
                break
    return model, log


@torch.no_grad()
def hidden_last(model, x):
    out = model(input_ids=x, output_hidden_states=True, use_cache=False)
    return [h[:, -1].float() for h in out.hidden_states], out.logits[:, -1, NV:NV + NVAL].float()


def probe_set(model, k, n, seed, shuffle=False):
    """Features at every layer (last position) + final logits; labels = hop-j variable for j = 1..k-1."""
    rng = random.Random(seed)
    feats, logs, labels = None, [], []
    for i in range(0, n, 500):
        seqs, chains = [], []
        for _ in range(min(500, n - i)):
            t, _, ch, _ = make_example(k, rng); seqs.append(t); chains.append(ch)
        hs, lg = hidden_last(model, batch_tensor(seqs).to(dev))
        feats = [h.cpu() for h in hs] if feats is None else [torch.cat([f, h.cpu()]) for f, h in zip(feats, hs)]
        logs.append(lg.cpu()); labels += [c[1:] for c in chains]
    y = torch.tensor(labels)                 # (n, k-1)
    if shuffle:
        y = y[torch.randperm(len(y))]
    return feats, torch.cat(logs), y


def fit_probe(Xtr, ytr, Xte, yte, classes=NV + NVAL + 6, steps=400, want_weights=False):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    Xtr, Xte = ((Xtr - mu) / sd).to(dev), ((Xte - mu) / sd).to(dev)
    W = torch.zeros(Xtr.shape[1], classes, device=dev, requires_grad=True)
    b = torch.zeros(classes, device=dev, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.05)
    ytr, yte = ytr.to(dev), yte.to(dev)
    for _ in range(steps):
        loss = F.cross_entropy(Xtr @ W + b, ytr) + 1e-4 * (W ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float(((Xte @ W + b).argmax(-1) == yte).float().mean())
    if want_weights:
        return acc, (W.detach() / sd.to(dev)[:, None])[:, :NV]          # raw-space directions for the 24 variables
    return acc


def probe_grid(model, k, seed, shuffle=False, extra=None):
    Ftr, Ltr, ytr = probe_set(model, k, N_PROBE_FIT, seed, shuffle)
    Fte, Lte, yte = probe_set(model, k, N_PROBE_TEST, seed + 1)
    res = {}
    for j in range(1, k):
        res[f"hop{j}"] = [fit_probe(Ftr[l], ytr[:, j - 1], Fte[l], yte[:, j - 1]) for l in range(len(Ftr))]
        res[f"hop{j}_from_final_value_logits"] = fit_probe(Ltr, ytr[:, j - 1], Lte, yte[:, j - 1])
        if extra is not None:
            for name, fn in extra.items():
                res[f"hop{j}_{name}"] = [fit_probe(fn(Ftr[l], l), ytr[:, j - 1], fn(Fte[l], l), yte[:, j - 1])
                                         for l in range(1, len(Ftr))]
    return res


def jlens_dirs(model, n=512, seed=7):
    """u_l(t) = mean over examples of d logit_t / d h_l at the last position, for the 32 task tokens."""
    rng = random.Random(seed)
    x, _ = gen_batch(n, rng)
    concepts = D
    U = {}
    model.eval()
    for l in range(1, LAYERS + 1):
        rows = []
        for t in concepts:
            store = {}

            def hook(_m, _i, o):
                h = o[0] if isinstance(o, tuple) else o
                h = h.detach().requires_grad_(True); store["h"] = h
                return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
            hk = model.gpt_neox.layers[l - 1].register_forward_hook(hook)
            lg = model(input_ids=x, use_cache=False).logits[:, -1, t].sum()
            hk.remove()
            g, = torch.autograd.grad(lg, store["h"])
            rows.append(g[:, -1].mean(0))
        U[l] = torch.stack(rows).float()          # (32, d)
    return U


def subspace_patch(model, k, seed):
    """Gate C (amended): same statements, two equal-depth chains. Swap ONLY the hop-j probe subspace of the last-position
    state at layer l from B into A; random subspace of equal rank as control. Switch = A's answer becomes B's."""
    rng = random.Random(seed)
    A, B, aB = [], [], []
    for _ in range(N_PATCH):
        t, ans, ch, other = make_example(k, rng, second_chain=True)
        ov, oval = other
        A.append(t); B.append(t[:-2] + [ov[0], COLON]); aB.append(oval - NV)
    xa, xb, ab = batch_tensor(A).to(dev), batch_tensor(B).to(dev), torch.tensor(aB, device=dev)
    Ftr, _, ytr = probe_set(model, k, N_PROBE_FIT, seed + 7)
    Fte, _, yte = probe_set(model, k, N_PROBE_TEST, seed + 8)
    with torch.no_grad():
        hsb = model(input_ids=xb, output_hidden_states=True, use_cache=False).hidden_states
        base = logits_last(model, xa).argmax(-1)
    res = {"baseline_switch": float((base == ab).float().mean())}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for j in range(1, k):
        prof, rprof, accs = [], [], []
        for l in range(1, LAYERS):
            acc, Wj = fit_probe(Ftr[l], ytr[:, j - 1], Fte[l], yte[:, j - 1], want_weights=True)
            Qm, _ = torch.linalg.qr(Wj.float())
            Qr, _ = torch.linalg.qr(torch.randn(Wj.shape, generator=g).to(dev))
            out = []
            for Qb in (Qm, Qr):
                def hook(_m, _i, o, Qb=Qb, src=hsb[l][:, -1]):
                    h = (o[0] if isinstance(o, tuple) else o).clone()
                    ha = h[:, -1].float()
                    h[:, -1] = (ha - (ha @ Qb) @ Qb.T + (src.float() @ Qb) @ Qb.T).to(h.dtype)
                    return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
                hk = model.gpt_neox.layers[l - 1].register_forward_hook(hook)
                with torch.no_grad():
                    pa = logits_last(model, xa).argmax(-1)
                hk.remove()
                out.append(float((pa == ab).float().mean()))
            prof.append(out[0]); rprof.append(out[1]); accs.append(acc)
        res[f"hop{j}"] = {"probe_acc_by_layer": accs, "switch_by_layer": prof, "random_switch_by_layer": rprof}
    return res


def main():
    t0 = time.time()
    teacher, log = train_teacher()
    OUT["teacher_log"] = log
    OUT["teacher_acc"] = {str(k): accuracy(teacher, k, 5000 if not TEST else 200, 1234) for k in ALL_K}
    print("teacher acc", OUT["teacher_acc"], f"({time.time() - t0:.0f}s)", flush=True); save()
    torch.save(teacher.state_dict(), "teacher4c.pt")

    U = jlens_dirs(teacher)
    W = teacher.get_output_embeddings().weight[NV:NV + NVAL].float()
    ln = teacher.gpt_neox.final_layer_norm
    extra = {"jlens8": lambda Xf, l: Xf.to(dev) @ U[l].T if l in U else Xf.to(dev)[:, :8],
             "logitlens8": lambda Xf, l: ln(Xf.to(dev)).float() @ W.T}
    def wrap(f):
        def g(Xf, l):
            with torch.no_grad():
                return f(Xf, l).detach().cpu()
        return g
    extra = {n: wrap(f) for n, f in extra.items()}
    OUT["gateA_probe_k5"] = probe_grid(teacher, 5, 100, extra=extra); save()
    print("gate A", {k: v for k, v in OUT["gateA_probe_k5"].items()}, flush=True)
    OUT["gateB_shuffled_labels_k5"] = probe_grid(teacher, 5, 200, shuffle=True)
    rand = new_model(12345)
    OUT["gateB_random_init_k5"] = probe_grid(rand, 5, 300); save()
    OUT["gateC_subspace"] = {str(k): subspace_patch(teacher, k, 400 + k) for k in (3, 5)}
    print("gate C", OUT["gateC_subspace"], f"({time.time() - t0:.0f}s)", flush=True)
    OUT["sec"] = round(time.time() - t0)
    save()


if __name__ == "__main__":
    main()
