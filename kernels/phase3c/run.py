"""Phase 2 pilot — install a big model's J-space in a small model while the small one is being born.

Teacher: Pythia-1.4B (24 layers, d=2048). Students: Pythia-70M-shaped models (6 layers, d=512) trained FROM ZERO on
the same text, in Pythia's own data order (first shard of the preshuffled Pile). Same tokenizer/vocabulary for all,
so concept scores live in the same word space and can be compared directly.

Students (identical data, steps, optimiser, seed unless noted):
  A_plain     ordinary next-word training                                     (the baseline)
  A_plain_s1  the same with another seed                                      (how big is plain noise?)
  B_jspace    ordinary training + "think the teacher's concepts": at its middle layer the student's concept scores
              must correlate with the teacher's J-lens concept scores at the teacher's middle layer, at every position
  C_kd        ordinary training + classic distillation (match the teacher's next-word probabilities)
  D_kd_jspace C + B together (does installing the J-space add anything on top of classic distillation?)

Report per student: held-out loss (lower = smarter at next word), J-lens edge at its middle layers (does it have a
workspace?), and teacher agreement (does its own J-lens, fitted afterwards, think the teacher's concepts?).
"""
import json
import math
import os
import re
import time

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM

TEST = os.environ.get("KAWKAB_TEST") == "1"
TEACHER = "EleutherAI/pythia-1.4b"
STUDENT_SHAPE = "EleutherAI/pythia-70m"
DATA = ("https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled/resolve/main/"
        "document-00000-of-00020.bin")
SEQ, BATCH, STEPS, WARMUP, LR = 256, 32, 6000, 300, 1e-3
T_LAYER, S_LAYER = 8, 2          # teacher layer 8 of 24, student layer 2 of 6 (both about 1/3 deep)
LAMBDA_J, TAU = 2.0, 2.0
N_CONCEPTS = 512
EVAL_SEQS = 200
# Phase 3a (after outside review): the KEY control. Paired runs, 3 arms per pair (same init + same data order within a
# pair; both change across pairs):
#   A  plain next-token training
#   B  + teacher target: a SEPARATE linear side head on student layer 2 (discarded after training; no final norm, no
#      unembedding, so the logit lens is untouched) trained to correlate with the teacher's layer-8 J-lens scores
#   T  + teacher-FREE target: the same side head trained on bag-of-future-words labels (which of the concepts appear
#      in the next 16 tokens) — same "extra homework", no teacher at all
# If T matches B, the gain is deep supervision, not the teacher's workspace. PAIRS selects which pairs this kernel runs.
PAIRS = [3, 4]
ARMS = ["A", "B", "T", "S"]
# S (added after pair 0): teacher-free but LEARNABLE target of matched form — the student's own final-layer logits on
# the same concepts (stop-grad). T's future-bag target was barely learnable (corr ~0.05), so S is the fairer control.
EVAL_EVERY = 1500
# Phase 2e: does the student USE the installed structure? Train A (plain) and B (install) from scratch exactly as in
# Phase 2, then at student layer 2 mean-ablate the k directions that best decode the teacher's J-content (ridge fit,
# each student its own) vs k random directions, and measure the eval-loss rise. If B leans on the installed
# directions (bigger rise than A, and than random), the installed workspace is functionally part of B.
ORDER = ["A_plain", "B_jspace"]
ABL_K = (4, 8, 16, 32)
N_RANDOM = 3
KD_MICRO = 4  # split each 32-sequence batch into 4 x 8 for the KD students (same tokens per step)
WORDS = ("time day year man woman child water food house city country war king money law school family music "
         "body death life world light fire tree river book game team door road car ship church army power heart "
         "mother father son daughter friend doctor police court market price company government president party "
         "black white red green blue cold hot fast slow good bad new old big small first last north south "
         "love fear hope pain help work play run walk speak write read sleep eat drink kill build open close "
         "number science history art language word name story question answer problem reason idea mind god "
         "village island mountain forest sea sun moon star night morning winter summer horse dog bird fish").split()
if TEST:
    SEQ, BATCH, STEPS, WARMUP, N_CONCEPTS, EVAL_SEQS, T_LAYER = 32, 4, 12, 2, 40, 8, 2

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
OUT = {"config": {"teacher": TEACHER, "student_shape": STUDENT_SHAPE, "seq": SEQ, "batch": BATCH, "steps": STEPS,
                  "lr": LR, "t_layer": T_LAYER, "s_layer": S_LAYER, "lambda_j": LAMBDA_J, "tau": TAU}, "students": {}}


def save():
    json.dump(OUT, open("phase3c.json", "w"), indent=1)


# ---------------------------------------------------------------- data (Pythia's own order, raw uint16, 2049/seq)
def fetch(start, nbytes):
    out, pos, step = bytearray(), start, 16 << 20
    while pos < start + nbytes:
        end = min(pos + step, start + nbytes) - 1
        for attempt in range(5):
            try:
                r = requests.get(DATA, headers={"Range": f"bytes={pos}-{end}"}, timeout=120)
                r.raise_for_status()
                break
            except Exception as ex:
                print("retry", attempt, ex, flush=True)
                time.sleep(5)
        out += r.content
        pos = end + 1
    return np.frombuffer(bytes(out), dtype=np.uint16)


def load_data():
    if TEST:
        rng = np.random.default_rng(0)
        return rng.integers(0, 500, size=(STEPS * BATCH * SEQ * 2,)), rng.integers(0, 500, size=(40000,))
    need = STEPS * BATCH * SEQ
    seqs = math.ceil(need / 2048) + 8
    train = fetch(0, seqs * 2049 * 2).reshape(-1, 2049)[:, :2048].reshape(-1)
    held = fetch(20_000_000_000, 2049 * 2 * 400).reshape(-1, 2049)[:, :2048].reshape(-1)  # ~10B tokens later
    return train, held


# ---------------------------------------------------------------- helpers
def find_layers(model):
    n = model.config.num_hidden_layers
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == n:
            return m
    raise RuntimeError("no layer list")


class Stop(Exception):
    pass


def layer_output(model, ids, layer, stop=True):
    """Hidden state after block `layer`; with stop=True the forward pass ends there (cheap)."""
    got = {}

    def hook(_m, _i, out):
        got["h"] = out[0] if isinstance(out, tuple) else out
        if stop:
            raise Stop

    h = find_layers(model)[layer].register_forward_hook(hook)
    try:
        model(input_ids=ids, use_cache=False)
    except Stop:
        pass
    finally:
        h.remove()
    return got["h"]


def auc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_jlens(model, fit, cids, layers_wanted, batch=32):
    """u_l(t) = J_l^T w_t with J_l = E[d h_final / d h_l] (sum over later positions, mean over source positions)."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    L = len(layers)
    W = model.get_output_embeddings().weight.float()
    acts = {}

    def keep(i):
        def hook(_m, _i, out):
            acts[i] = out[0] if isinstance(out, tuple) else out
        return hook

    hooks = [layers[i].register_forward_hook(keep(i)) for i in set(layers_wanted) | {L - 1}]
    hooks.append(model.get_input_embeddings().register_forward_hook(lambda _m, _i, o: o.detach().requires_grad_(True)))
    seq = len(fit[0])
    valid = torch.zeros(seq, dtype=torch.bool, device=dev)
    valid[8:seq - 1] = True
    U = {i: torch.zeros(len(cids), W.shape[1], device=dev) for i in layers_wanted}
    for ids in fit:
        for b in range(0, len(cids), batch):
            chunk = cids[b:b + batch]
            acts.clear()
            model(input_ids=torch.tensor([ids] * len(chunk), device=dev), use_cache=False)
            final = acts[L - 1]
            cot = torch.zeros_like(final)
            cot[:, valid, :] = W[chunk][:, None, :].to(final.dtype)
            grads = torch.autograd.grad(final, [acts[i] for i in layers_wanted], grad_outputs=cot, allow_unused=True)
            for li, g in zip(layers_wanted, grads):
                if g is not None:
                    U[li][b:b + len(chunk)] += g[:, valid, :].float().mean(1)
    for h in hooks:
        h.remove()
    return {i: u / len(fit) for i, u in U.items()}


def zrows(x):
    x = x - x.mean(-1, keepdim=True)
    return x / (x.norm(dim=-1, keepdim=True) + 1e-6)


# ---------------------------------------------------------------- teacher
def load_teacher(tok, fit):
    if TEST:
        cfg = shape(4, 64)
        teacher = GPTNeoXForCausalLM(cfg).to(dev)
    else:
        teacher = AutoModelForCausalLM.from_pretrained(TEACHER, torch_dtype=torch.float32).to(dev)
    return teacher


def concepts(tok, train):
    """The most frequent whole-word tokens in the training text (skip the 40 commonest = function words) + WORDS."""
    counts = np.bincount(train[: 20_000_000].astype(np.int64), minlength=len(tok))
    cands = [i for i in np.argsort(-counts) if i < len(tok) and counts[i] > 0
             and re.fullmatch(r" [a-z]{4,}", tok.decode([int(i)]) or "")]
    cids = [int(i) for i in cands[40:40 + N_CONCEPTS]] if not TEST else [int(i) for i in cands[:N_CONCEPTS]]
    words = []
    for w in WORDS:
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            words.append(t[0])
    return sorted(set(cids) | set(words)), sorted(set(words))


# ---------------------------------------------------------------- students
def shape(layers, d):
    if TEST:
        from transformers import GPTNeoXConfig
        return GPTNeoXConfig(vocab_size=512, hidden_size=d, num_hidden_layers=layers, num_attention_heads=4,
                             intermediate_size=4 * d, max_position_embeddings=256)
    return AutoConfig.from_pretrained(STUDENT_SHAPE)


def new_student(seed):
    torch.manual_seed(seed)
    cfg = shape(6, 32)
    return GPTNeoXForCausalLM(cfg).to(dev)


def batches(train, seed):
    n = len(train) // SEQ
    order = np.random.default_rng(1000 + seed).permutation(n)  # data order: same within a pair, differs across pairs
    for s in range(STEPS):
        idx = order[(s * BATCH) % n:(s * BATCH) % n + BATCH]
        x = np.stack([train[i * SEQ:(i + 1) * SEQ] for i in idx]).astype(np.int64)
        yield torch.from_numpy(x).to(dev)


def future_bag(x, tok2c, C, horizon=16):
    """(b, SEQ, C) 1 where concept c appears among the next `horizon` tokens."""
    b, n = x.shape
    out = torch.zeros(b, n, C, device=x.device)
    for k in range(1, horizon + 1):
        c = tok2c[x[:, k:]]
        oh = F.one_hot(c.clamp(min=0), C).float() * (c >= 0).unsqueeze(-1).float()
        out[:, : n - k] = torch.maximum(out[:, : n - k], oh)
    return out


def train_student(name, train, teacher, Ut, cids_t, arm, seed, held):
    student = new_student(seed)
    student.train()
    C = len(cids_t)
    torch.manual_seed(10_000 + seed)
    head = torch.nn.Linear(student.config.hidden_size, C).to(dev) if arm in ("B", "T", "S") else None
    tok2c = torch.full((student.config.vocab_size,), -1, dtype=torch.long, device=dev)
    tok2c[torch.tensor(cids_t, device=dev)] = torch.arange(C, device=dev)
    cset_t = torch.tensor(cids_t, device=dev)
    params = list(student.parameters()) + (list(head.parameters()) if head is not None else [])
    use_kd = False
    curve = []
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1, s / STEPS)))))
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    log, t0 = [], time.time()
    acts = {}
    hook = find_layers(student)[S_LAYER].register_forward_hook(
        lambda _m, _i, o: acts.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    micro = KD_MICRO if use_kd else 1
    for step, xb in enumerate(batches(train, seed)):
      opt.zero_grad(set_to_none=True)
      for x in xb.chunk(micro):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            out = student(input_ids=x, use_cache=False)
            logits = out.logits[:, :-1].float()
            ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1))
            loss, jl, kd = ce, torch.tensor(0.0), torch.tensor(0.0)
            if head is not None:
                ss = zrows(head(acts["h"][:, 8:].float()))
                if arm == "B":
                    with torch.no_grad():
                        st = zrows(layer_output(teacher, x, T_LAYER)[:, 8:].float() @ Ut.T)
                elif arm == "S":
                    st = zrows(out.logits[:, 8:][..., cset_t].detach().float())
                else:
                    st = zrows(future_bag(x, tok2c, C)[:, 8:])
                jl = 1 - (ss * st).sum(-1).mean()
                loss = loss + LAMBDA_J * jl
            if False:
                with torch.no_grad():
                    if use_kd:
                        th = {}
                        hk = find_layers(teacher)[T_LAYER].register_forward_hook(
                            lambda _m, _i, o: th.__setitem__("h", o[0] if isinstance(o, tuple) else o))
                        tlogits = teacher(input_ids=x, use_cache=False).logits[:, :-1]
                        hk.remove()
                        ht = th["h"]
                    else:
                        ht = layer_output(teacher, x, T_LAYER)
                    st = zrows(ht[:, 8:].float() @ Ut.T)  # teacher's J-space content per position
                if use_kd:
                    kd = F.kl_div(F.log_softmax(logits / TAU, -1).reshape(-1, logits.shape[-1]),
                                  F.log_softmax(tlogits / TAU, -1).reshape(-1, logits.shape[-1]),
                                  log_target=True, reduction="batchmean") * TAU ** 2
                    loss = 0.5 * loss + 0.5 * kd   # (KD unused in 3a)
        scaler.scale(loss / micro).backward()
        del out, logits
        if use_kd:
            del tlogits
      if True:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % 250 == 0 or step == STEPS - 1:
            row = {"step": step, "tokens_M": round((step + 1) * BATCH * SEQ / 1e6, 1), "ce": round(ce.item(), 4),
                   "j_loss": round(float(jl), 4), "kd": round(float(kd), 4), "sec": round(time.time() - t0)}
            log.append(row)
            print(name, row, flush=True)
        if (step + 1) % EVAL_EVERY == 0:
            curve.append({"step": step + 1, "eval_loss": eval_loss(student, held)})
            student.train()
    hook.remove()
    return student, log, curve


@torch.no_grad()
def split_loss(model, held, cids_t):
    """Eval loss on target tokens inside the concept set vs outside it."""
    model.eval()
    x = torch.from_numpy(held[: EVAL_SEQS * SEQ].reshape(EVAL_SEQS, SEQ).astype(np.int64)).to(dev)
    inset = torch.zeros(model.config.vocab_size, dtype=torch.bool, device=dev)
    inset[torch.tensor(cids_t, device=dev)] = True
    s_in = s_out = n_in = n_out = 0.0
    for i in range(0, EVAL_SEQS, 16):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = model(input_ids=x[i:i + 16], use_cache=False).logits[:, :-1].float()
        y = x[i:i + 16, 1:]
        l = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none")
        m = inset[y.reshape(-1)]
        s_in += l[m].sum().item(); n_in += m.sum().item()
        s_out += l[~m].sum().item(); n_out += (~m).sum().item()
    return {"in_set": s_in / max(n_in, 1), "out_set": s_out / max(n_out, 1), "frac_in": n_in / (n_in + n_out)}


@torch.no_grad()
def eval_loss(model, held):
    model.eval()
    x = torch.from_numpy(held[: EVAL_SEQS * SEQ].reshape(EVAL_SEQS, SEQ).astype(np.int64)).to(dev)
    tot = 0.0
    for i in range(0, EVAL_SEQS, 16):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            lg = model(input_ids=x[i:i + 16], use_cache=False).logits[:, :-1].float()
        tot += F.cross_entropy(lg.reshape(-1, lg.shape[-1]), x[i:i + 16, 1:].reshape(-1), reduction="sum").item()
    return tot / (EVAL_SEQS * (SEQ - 1))


def windows(held, n, length=64, start=0):
    return [held[start + i * length: start + (i + 1) * length].astype(np.int64).tolist() for i in range(n)]


def workspace_report(student, teacher, Ut, cids_t, cids_w, held):
    """J-lens edge (phase 1 measure, WORDS concepts) at every student layer + agreement with the teacher's J-space."""
    fit = windows(held, 48 if not TEST else 4, start=EVAL_SEQS * SEQ)
    ev = windows(held, 32 if not TEST else 4, start=EVAL_SEQS * SEQ + 48 * 64)
    L = student.config.num_hidden_layers
    Us = fit_jlens(student, fit, cids_w, list(range(L - 1)))
    Us_t = fit_jlens(student, fit, cids_t, [S_LAYER])[S_LAYER]
    W = student.get_output_embeddings().weight.float()
    cset = torch.tensor(cids_w, device=dev)
    rows = []
    with torch.no_grad():
        acts = {}
        hooks = [find_layers(student)[i].register_forward_hook(
            (lambda i: lambda _m, _in, o: acts.__setitem__(i, o[0] if isinstance(o, tuple) else o))(i)) for i in range(L)]
        H, labels, agree_j, agree_logit = {i: [] for i in range(L - 1)}, [], [], []
        for ids in ev:
            x = torch.tensor([ids], device=dev)
            acts.clear()
            student(input_ids=x, use_cache=False)
            for p in range(8, 64 - 16):
                labels.append(torch.isin(cset, x[0, p + 1:p + 17]).cpu().numpy())
                for i in range(L - 1):
                    H[i].append(acts[i][0, p].float())
            ht = layer_output(teacher, x, T_LAYER)[0, 8:].float()
            st = zrows(ht @ Ut.T)
            hs = acts[S_LAYER][0, 8:].float()
            agree_j.append((zrows(hs @ Us_t.T) * st).sum(-1).mean().item())
            hl = student.gpt_neox.final_layer_norm(acts[S_LAYER][0, 8:]).float()
            agree_logit.append((zrows(hl @ W[cids_t].T) * st).sum(-1).mean().item())
        for h in hooks:
            h.remove()
        labels = np.stack(labels)
        for i in range(L - 1):
            Hi = torch.stack(H[i])
            sj = (Hi @ Us[i].T).cpu().numpy()
            sl = (Hi @ W[cids_w].T).cpu().numpy()
            sj, sl = sj - sj.mean(1, keepdims=True), sl - sl.mean(1, keepdims=True)
            rows.append({"layer": i, "auc_jlens": auc(sj.ravel(), labels.ravel()),
                         "auc_logit": auc(sl.ravel(), labels.ravel())})
    return {"layers": rows, "teacher_agreement_jlens": float(np.mean(agree_j)),
            "teacher_agreement_readout": float(np.mean(agree_logit))}


@torch.no_grad()
def ablation_report(student, teacher, Ut, held):
    """Fit ridge H(layer S_LAYER) -> teacher J-scores; ablate its top-k input directions vs random ones."""
    student.eval()
    fit = windows(held, 96 if not TEST else 6, length=SEQ, start=EVAL_SEQS * SEQ + 20000)
    Hs, Ts = [], []
    for ids in fit:
        x = torch.tensor([ids], device=dev)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
            hs = layer_output(student, x, S_LAYER)[0, 8:].float()
            ht = layer_output(teacher, x, T_LAYER)[0, 8:].float()
        Hs.append(hs)
        Ts.append(zrows(ht @ Ut.T))
    H, T = torch.cat(Hs), torch.cat(Ts)
    mu = H.mean(0)
    Hc, Tc = H - mu, T - T.mean(0)
    n_tr = int(0.8 * len(Hc))
    d = Hc.shape[1]
    lam = 1e-2 * (Hc[:n_tr] ** 2).sum() / n_tr
    M = torch.linalg.solve(Hc[:n_tr].T @ Hc[:n_tr] + lam * torch.eye(d, device=dev), Hc[:n_tr].T @ Tc[:n_tr])
    pred = Hc[n_tr:] @ M
    r2 = float(1 - ((Tc[n_tr:] - pred) ** 2).sum() / (Tc[n_tr:] ** 2).sum())
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    Sigma = Hc.T @ Hc / len(Hc)
    Uw, _, _ = torch.linalg.svd(Sigma @ M, full_matrices=False)       # variance-weighted aligned directions
    evals, evecs = torch.linalg.eigh(Sigma)
    Up = evecs[:, torch.argsort(evals, descending=True)]               # PCA, max-variance reference
    tot = float(torch.trace(Sigma))
    sets = {"aligned": U, "aligned_var": Uw, "pca": Up}
    out = {"decode_r2": r2, "base_loss": eval_loss(student, held), "random": {}, "var_frac": {}}
    for nm in sets:
        out[nm] = {}
    layer = find_layers(student)[S_LAYER]

    def loss_without(P):
        def hook(_m, _i, o):
            h = o[0] if isinstance(o, tuple) else o
            h2 = h - ((h.float() - mu) @ P @ P.T).to(h.dtype)
            return (h2,) + tuple(o[1:]) if isinstance(o, tuple) else h2
        hk = layer.register_forward_hook(hook)
        try:
            return eval_loss(student, held)
        finally:
            hk.remove()

    g = torch.Generator(device="cpu").manual_seed(0)
    vf = lambda P: float(torch.trace(P.T @ Sigma @ P)) / tot
    for k in ABL_K:
        k = min(k, d)
        out["var_frac"][k] = {}
        for nm, B in sets.items():
            P = B[:, :k].contiguous()
            out[nm][k] = loss_without(P) - out["base_loss"]
            out["var_frac"][k][nm] = vf(P)
        rs, vs = [], []
        for _ in range(N_RANDOM):
            Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g).to(dev))
            rs.append(loss_without(Q) - out["base_loss"])
            vs.append(vf(Q))
        out["random"][k] = float(np.mean(rs))
        out["var_frac"][k]["random"] = float(np.mean(vs))
    return out


class FakeTok:
    """Offline stand-in for the test run only."""
    words = ["time", "river", "king", "water"] + [f"w{i}" for i in range(508)]

    def __len__(self):
        return 512

    def decode(self, ids):
        return "".join(" " + (self.words[i] if i < 4 else "abcd" + "efghijklmnopqrstuvwxyz"[i % 22]) for i in ids)

    def __call__(self, text, add_special_tokens=False):
        w = text.strip()
        return {"input_ids": [self.words.index(w)] if w in self.words[:4] else [1, 2]}


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(TEACHER) if not TEST else FakeTok()
    train, held = load_data()
    print("data", train.shape, held.shape, "max token", int(train.max()), flush=True)
    if not TEST:
        print("first text:", repr(tok.decode(train[:60].tolist())), flush=True)
    cids_t, cids_w = concepts(tok, train)
    OUT["concepts"] = len(cids_t)
    print("concepts", len(cids_t), [tok.decode([c]) for c in cids_t[:30]], flush=True)
    teacher = load_teacher(tok, None)
    fit_t = windows(held, 48 if not TEST else 4, start=EVAL_SEQS * SEQ + 200 * 64)
    Ut = fit_jlens(teacher, fit_t, cids_t, [T_LAYER])[T_LAYER]
    if dev == "cuda":
        teacher = teacher.half()
    Ut = Ut.to(dev)
    OUT["teacher_eval_loss"] = eval_loss(teacher, held)
    print("teacher J-lens fitted, eval loss", OUT["teacher_eval_loss"], f"({time.time() - t0:.0f}s)", flush=True)
    save()
    OUT["pairs"] = PAIRS
    for pair in PAIRS:
        for arm in ARMS:
            name = f"{arm}_p{pair}"
            try:
                student, log, curve = train_student(name, train, teacher, Ut, cids_t, arm, pair, held)
                res = {"arm": arm, "pair": pair, "log": log, "curve": curve, "eval_loss": eval_loss(student, held)}
                res["split"] = split_loss(student, held, cids_t)
                print(name, "eval loss", round(res["eval_loss"], 4), "split", res["split"],
                      f"({time.time() - t0:.0f}s total)", flush=True)
                del student
            except Exception as ex:
                import traceback
                traceback.print_exc()
                res = {"error": str(ex)[:500]}
            OUT["students"][name] = res
            save()
            if dev == "cuda":
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

