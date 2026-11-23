# Phase 5a (PREREGISTRATION-phase5.md): W1 = W switched off after curriculum stage 1. Same code/budget/lambda as 4f/4g.
# Phase 4g = Phase 4f continued (seeds 2 rest + 3), frozen budget/lambdas from 4f. Phase 4f — AMENDMENT 3 (EXPLORATORY transfer test). Load the Phase 4e teacher (sha256-checked), re-sited gates A'/B'
# at statement RHS positions, then a pilot (budget + loss weights), then paired student seeds for arms
# A plain, K output distillation, W teacher hidden states (all non-pad positions, teacher L2/L5/L7 -> student L2/L3/L5),
# X W from another example, R random-init teacher, O oracle resolved-value labels, Wm W with probe directions removed.
import json, os, random, time
import torch
import torch.nn.functional as F
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

import hashlib, urllib.request
import torch.nn as nn

TEST = os.environ.get("KAWKAB_TEST") == "1"
dev = "cuda" if torch.cuda.is_available() else "cpu"
PHASE = "5a"
RUNS = [("W1", 1), ("W1", 2), ("W1", 3), ("W1", 4), ("W1", 5), ("W", 4), ("W", 5), ("A", 4), ("A", 5)]   # W/A seeds 1-3 exist (4f/4g)
FROZEN = {"budget": 20000, "lam": {"W": 1.0, "O": 0.1, "K": 0.1}}   # copied from phase4f.json (pilot seed 100)
OUT = {"phase": PHASE, "amendment": 3, "runs": {}}

NV, NVAL, NSTMT = 24, 8, 12
TRAIN_K, ALL_K = (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 6, 7)
K_WEIGHTS = (0.4 / 3, 0.4 / 3, 0.4 / 3, 0.3, 0.3)
P_NEXT = 0.0
LAYERS, WIDTH, HEADS = 8, 256, 8                     # teacher (fixed: loaded from Phase 4e)
S_LAYERS, S_WIDTH, S_HEADS = (6, 128, 4) if not TEST else (6, 32, 4)
T_SITES, S_SITES = (2, 5, 7), (2, 3, 5)
BATCH, LR = (512, 1e-3) if not TEST else (32, 1e-3)
BUDGET_CAP = 20000 if not TEST else 60
POOL = 2_000_000 if not TEST else 4000
EVAL_EVERY = 1000 if not TEST else 20
N_EVAL = 2000 if not TEST else 100
LAMS = (0.1, 0.3, 1.0)
ARMS = ("A", "K", "W", "X", "R", "O", "Wm")
TIME_LIMIT = 11 * 3600
TEACHER_URL = "https://raw.githubusercontent.com/Afande90/kawkab-runner/main/results/phase4e/teacher4e.pt"
TEACHER_SHA = "61172681c0a3932826e2b217d95b4c86793ad596822ff2d18e8615a017619717"

V = list(range(NV)); D = list(range(NV, NV + NVAL))
EQ, SEMI, Q, COLON, NEXT, PAD = range(NV + NVAL, NV + NVAL + 6)
VOCAB = NV + NVAL + 6
T0 = time.time()


def save():
    json.dump(OUT, open(f"phase{PHASE}.json", "w"), indent=1)


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


def rhs_records(t):
    """For each statement: (RHS token index, resolved value 0..7, hops d, all links earlier?)"""
    st = {t[p]: (t[p + 2], p) for p in range(0, len(t) - 3, 4) if t[p + 1] == EQ}
    out = []
    for v, (rhs, p) in st.items():
        d, cur, back = 1, rhs, True
        while cur < NV and cur in st and d < 12:
            if st[cur][1] > p:
                back = False
            cur = st[cur][0]; d += 1
        if cur >= NV:
            out.append((p + 2, cur - NV, d, back))
    return out


def _pool_part(args):
    n, seed = args
    rng = random.Random(seed)
    seqs, ans, ks, labs = [], [], [], []
    for _ in range(n):
        k = TRAIN_K[len(seqs) % len(TRAIN_K)]
        t, a, _, _ = make_example(k, rng)
        lab = [-1] * len(t)
        for pos, val, d, back in rhs_records(t):
            if back:
                lab[pos] = val
        seqs.append(t); ans.append(a); ks.append(k); labs.append(lab)
    n_ = max(len(s) for s in seqs)
    lt = torch.full((len(labs), n_), -1, dtype=torch.long)
    for i, l in enumerate(labs):
        lt[i, n_ - len(l):] = torch.tensor(l)
    return batch_tensor(seqs), torch.tensor(ans), torch.tensor(ks), lt


def build_pool():
    import multiprocessing as mp
    parts = max(1, os.cpu_count() or 1)
    with mp.Pool(parts) as p:
        res = p.map(_pool_part, [(POOL // (parts * 8), 1000 + i) for i in range(parts * 8)])
    n = max(r[0].shape[1] for r in res)
    xs = torch.cat([torch.cat([torch.full((r[0].shape[0], n - r[0].shape[1]), PAD, dtype=torch.long), r[0]], 1) for r in res])
    ls = torch.cat([torch.cat([torch.full((r[3].shape[0], n - r[3].shape[1]), -1, dtype=torch.long), r[3]], 1) for r in res])
    return xs.to(dev), torch.cat([r[1] for r in res]).to(dev), torch.cat([r[2] for r in res]).to(dev), ls.to(dev)


def new_model(seed, layers=LAYERS, width=WIDTH, heads=HEADS):
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(vocab_size=VOCAB, hidden_size=width, num_hidden_layers=layers, num_attention_heads=heads,
                        intermediate_size=4 * width, max_position_embeddings=128, rotary_pct=0.25,
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




def load_teacher():
    path = "teacher4e.pt"
    for cand in ("teacher4e.pt", "../../results/phase4e/teacher4e.pt"):
        if os.path.exists(cand):
            path = cand; break
    else:
        urllib.request.urlretrieve(TEACHER_URL, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert sha == TEACHER_SHA, "teacher file mismatch " + sha
    t = new_model(0)
    t.load_state_dict(torch.load(path, map_location=dev)); t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


@torch.no_grad()
def states(model, x, sites=T_SITES):
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
        out = model(input_ids=x, output_hidden_states=True, use_cache=False)
    return [out.hidden_states[l].float() for l in sites], out.logits[:, -1, NV:NV + NVAL].float()


def fit_linear(Xtr, ytr, Xte, yte, classes=NVAL, steps=400):
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
    return acc, (W.detach() / sd.to(dev)[:, None])


def rhs_feature_set(model, n, seed):
    """Teacher features at RHS tokens of all-earlier statements, at T_SITES; labels and hop counts."""
    rng = random.Random(seed)
    feats, ys, ds = [[] for _ in T_SITES], [], []
    for i in range(0, n, 250):
        exs = [make_example(5, rng) for _ in range(min(250, n - i))]
        x = batch_tensor([e[0] for e in exs]).to(dev); L = x.shape[1]
        hs, _ = states(model, x)
        for j, (t, _, _, _) in enumerate(exs):
            for pos, val, d, back in rhs_records(t):
                if back and d >= 2:
                    for s in range(len(T_SITES)):
                        feats[s].append(hs[s][j, L - len(t) + pos].cpu())
                    ys.append(val); ds.append(d)
    return [torch.stack(f) for f in feats], torch.tensor(ys), torch.tensor(ds)


def gates(teacher, rand):
    n = 6000 if not TEST else 200
    res = {}
    F_, y, d = rhs_feature_set(teacher, n, 11)
    Fr, yr, dr = rhs_feature_set(rand, n, 11)
    dirs = []
    for s, l in enumerate(T_SITES):
        cut = int(len(y) * 0.8)
        row = {}
        for dd in (2, 3, 4):
            m = d == dd; idx = m.nonzero().flatten(); c = int(len(idx) * 0.8)
            if len(idx) < 50:
                continue
            row[f"d{dd}"] = round(fit_linear(F_[s][idx[:c]], y[idx[:c]], F_[s][idx[c:]], y[idx[c:]])[0], 3)
        acc, W = fit_linear(F_[s][:cut], y[:cut], F_[s][cut:], y[cut:])
        row["all"] = round(acc, 3); dirs.append(W)
        perm = y[torch.randperm(len(y))]
        row["shuffled"] = round(fit_linear(F_[s][:cut], perm[:cut], F_[s][cut:], perm[cut:])[0], 3)
        c2 = int(len(yr) * 0.8)
        row["random_init"] = round(fit_linear(Fr[s][:c2], yr[:c2], Fr[s][c2:], yr[c2:])[0], 3)
        res[f"L{l}"] = row
    last = res[f"L{T_SITES[-1]}"]
    ok_a = all(last.get(k, 0) >= 0.85 for k in ("d2", "d3"))
    ok_b = all(res[f"L{l}"]["shuffled"] <= 0.25 for l in T_SITES) and \
        last["all"] - last["random_init"] >= 0.35          # Amendment 3a: random-init floor ~0.5 is the task's base rate
    res["pass_A"], res["pass_B"] = ok_a, ok_b
    return res, dirs


@torch.no_grad()
def pca_targets(model, px, proj=None, n_batches=40):
    """Per transfer layer: mean, sd, projection to top PCs of z-scored states (95% var), scale; proj = optional (Q) to remove first."""
    g = torch.Generator(device=dev).manual_seed(5)
    out = []
    acc = [[] for _ in T_SITES]
    for _ in range(n_batches if not TEST else 4):
        x = px[torch.randint(0, len(px), (256 if not TEST else 32,), device=dev, generator=g)]
        hs, _ = states(model, x)
        m = x != PAD
        for s in range(len(T_SITES)):
            h = hs[s][m]
            if proj is not None:
                h = h - (h @ proj[s]) @ proj[s].T
            acc[s].append(h)
    for s in range(len(T_SITES)):
        H = torch.cat(acc[s]); mu = H.mean(0); sd = H.std(0) + 1e-5      # Amendment 3a: z-score dims before PCA
        ev, evec = torch.linalg.eigh(torch.cov(((H - mu) / sd).T))
        ev, evec = ev.flip(0).clamp_min(1e-8), evec.flip(1)
        k = int((ev.cumsum(0) / ev.sum() < 0.95).sum().item()) + 1
        out.append((mu, sd, evec[:, :k], ev[:k].sqrt()))
    return out


def make_target(arm, x, teacher, rand, pcs, proj):
    src = rand if arm == "R" else teacher
    hs, tl = states(src, x)
    if arm == "K":
        return None, None, tl
    tg = []
    for s in range(len(T_SITES)):
        h = hs[s]
        if arm == "Wm":
            h = h - (h @ proj[s]) @ proj[s].T
        mu, sd, P, sc = pcs[s]
        tg.append((((h - mu) / sd) @ P) / sc)
    m = x != PAD
    if arm == "X":
        tg = [t.roll(1, 0) for t in tg]; m = m & m.roll(1, 0)
    return tg, m, tl


def train_student(arm, seed, lam, budget, data, teacher, rand, pcs_all, proj, stop_at=None):
    px, pa, pk, pl = data
    by_k = {k: (pk == k).nonzero().flatten() for k in TRAIN_K}
    model = new_model(seed, S_LAYERS, S_WIDTH, S_HEADS)
    heads = nn.ModuleList()
    if arm in ("W", "W1", "X", "R", "Wm"):
        heads = nn.ModuleList([nn.Linear(S_WIDTH, pcs_all[arm][s][2].shape[1]) for s in range(len(S_SITES))]).to(dev)
    elif arm == "O":
        heads = nn.ModuleList([nn.Linear(S_WIDTH, NVAL) for _ in S_SITES]).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(heads.parameters()), lr=LR, weight_decay=0.1, betas=(0.9, 0.98))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (500 if not TEST else 5)))
    gen = torch.Generator(device=dev).manual_seed(1000 + seed)          # same data order for every arm of a seed
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    def stage_weights(Dd):
        if Dd > len(TRAIN_K):
            return torch.tensor(K_WEIGHTS, device=dev)
        w = torch.zeros(len(TRAIN_K), device=dev)
        if Dd == 1:
            w[0] = 1.0
        else:
            w[:Dd - 1] = 0.5 / (Dd - 1); w[Dd - 1] = 0.5
        return w
    Dd, curve, to90, aux = 1, [], None, torch.zeros(())
    for step in range(budget):
        model.train()
        kk = torch.multinomial(stage_weights(Dd), BATCH, replacement=True, generator=gen)
        idx = torch.empty(BATCH, dtype=torch.long, device=dev)
        for i, k in enumerate(TRAIN_K):
            msk = kk == i
            if msk.any():
                pool_k = by_k[k]
                idx[msk] = pool_k[torch.randint(0, len(pool_k), (int(msk.sum()),), device=dev, generator=gen)]
        x, a, lab = px[idx], pa[idx], pl[idx]
        x = x[:, (x != PAD).any(0).nonzero().min():]; lab = lab[:, -x.shape[1]:]
        tg = None
        eff = "A" if (arm == "W1" and Dd > 1) else ("W" if arm == "W1" else arm)    # W1: transfer only in stage 1
        if eff in ("W", "X", "R", "Wm", "K"):
            tg, m, tl = make_target(eff, x, teacher, rand, pcs_all.get(eff), proj)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = model(input_ids=x, output_hidden_states=eff not in ("A", "K"), use_cache=False)
            lg = out.logits[:, -1, NV:NV + NVAL].float()
            loss = F.cross_entropy(lg, a - NV)
            if eff in ("W", "X", "R", "Wm"):
                aux = sum(F.mse_loss(heads[s](out.hidden_states[l][m]).float(), tg[s][m]) for s, l in enumerate(S_SITES)) / len(S_SITES)
                loss = loss + lam * aux
            elif eff == "O":
                mm = lab >= 0
                aux = sum(F.cross_entropy(heads[s](out.hidden_states[l][mm]).float(), lab[mm]) for s, l in enumerate(S_SITES)) / len(S_SITES)
                loss = loss + lam * aux
            elif eff == "K":
                loss = loss + lam * F.kl_div(F.log_softmax(lg, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean")
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(heads.parameters()), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        if (step + 1) % EVAL_EVERY == 0:
            acc = {k: accuracy(model, k, N_EVAL, 99) for k in TRAIN_K}
            curve.append({"step": step + 1, "stage": Dd, "acc": acc, "loss": round(loss.item(), 4), "aux": round(float(aux), 4)})
            if to90 is None and acc[5] >= 0.9:
                to90 = step + 1
            if Dd <= len(TRAIN_K) and all(acc[k] >= 0.99 for k in TRAIN_K[:Dd]):
                Dd += 1
            if stop_at is not None and acc[5] >= stop_at:
                break
    final = {str(k): accuracy(model, k, 5000 if not TEST else 100, 4321) for k in ALL_K}
    return {"final": final, "primary": round((final["4"] + final["5"]) / 2, 4), "steps_to_90_k5": to90,
            "curve": curve, "steps": step + 1}


def main():
    data = build_pool()                       # before any torch compute (fork safety)
    print("pool", tuple(data[0].shape), f"({time.time() - T0:.0f}s)", flush=True)
    teacher = load_teacher()
    rand = new_model(12345); rand.eval()
    OUT["teacher_acc"] = {str(k): accuracy(teacher, k, 2000 if not TEST else 100, 1234) for k in (4, 5)}
    g, dirs = gates(teacher, rand)
    OUT["gates"] = g; save(); print("gates", g, flush=True)
    if not (g["pass_A"] and g["pass_B"]) and not TEST:
        OUT["stopped"] = "gate A' or B' failed: no students"; save(); return
    proj = [torch.linalg.qr(W)[0] for W in dirs]
    pcs_t = pca_targets(teacher, data[0]); pcs_r = pca_targets(rand, data[0]); pcs_m = pca_targets(teacher, data[0], proj)
    pcs_all = {"W": pcs_t, "W1": pcs_t, "X": pcs_t, "R": pcs_r, "Wm": pcs_m}
    OUT["n_pcs"] = {a: [p[2].shape[1] for p in v] for a, v in pcs_all.items()}
    run = lambda arm, seed, lam, budget, **kw: train_student(arm, seed, lam, budget, data, teacher, rand, pcs_all, proj, **kw)

    if FROZEN is None:
        p = run("A", 100, 0.0, BUDGET_CAP, stop_at=0.5)
        budget = p["steps"]
        OUT["pilot"] = {"A": p, "budget": budget}; save()
        print("pilot budget", budget, p["final"], f"({time.time() - T0:.0f}s)", flush=True)
        lam = {}
        for arm in ("W", "O", "K"):
            sc = {}
            for l_ in LAMS:
                r = run(arm, 100, l_, budget); sc[str(l_)] = r["primary"]
                OUT["pilot"][f"{arm}_{l_}"] = r["final"]; save()
            lam[arm] = float(max(sc, key=sc.get))
            print("pilot", arm, sc, f"({time.time() - T0:.0f}s)", flush=True)
        frozen = {"budget": budget, "lam": lam}
    else:
        frozen = FROZEN
    if TEST:
        frozen = {**frozen, "budget": BUDGET_CAP}
    OUT["frozen"] = frozen; save()
    lam_of = {"A": 0.0, "K": frozen["lam"]["K"], "O": frozen["lam"]["O"]}
    for arm, seed in RUNS:
        if time.time() - T0 > TIME_LIMIT:
            OUT["stopped"] = "time limit"; save(); return
        r = run(arm, seed, lam_of.get(arm, frozen["lam"]["W"]), frozen["budget"])
        OUT["runs"][f"s{seed}_{arm}"] = r; save()
        print("seed", seed, arm, r["primary"], r["final"], f"({time.time() - T0:.0f}s)", flush=True)
    OUT["sec"] = round(time.time() - T0); save()


if __name__ == "__main__":
    main()
