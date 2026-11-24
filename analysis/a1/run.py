# Analysis a1 — measuring checks on public Pythia models (no training, no GPU).
# For each size: (1) board size, (2) content quality vs number of directions kept, (3) writers, (4) readers,
# (5) cross-size similarity of concept geometry, (6) how much of the largest model's board content a smaller
# model's state already holds (ridge), against token-identity and random-init baselines.
import os, json, re, time, math
import numpy as np, torch, requests
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

TEST = os.environ.get("KAWKAB_TEST") == "1"
torch.set_num_threads(os.cpu_count() or 2)
SIZES = ["pythia-14m", "pythia-31m", "pythia-70m", "pythia-160m", "pythia-410m"]
BIG = "pythia-410m"
DATA = ("https://huggingface.co/datasets/EleutherAI/pile-standard-pythia-preshuffled/resolve/main/"
        "document-00000-of-00020.bin")
SEQ, N_FIT, N_EVAL, N_CONC, CHUNK, H = 128, 12, 64, 256, 64, 16
KS = (2, 4, 8, 16, 32, 64, 128)
WORDS = ("time day year man woman child water food house city country war king money law school family music "
         "body death life world light fire tree river book game team door road car ship church army power heart "
         "mother father son daughter friend doctor police court market price company government president party "
         "number science history art language word name story question answer problem reason idea mind god").split()
if TEST:
    SIZES, BIG = ["t-a", "t-b"], "t-b"
    SEQ, N_FIT, N_EVAL, N_CONC, CHUNK, H = 24, 2, 4, 20, 10, 4
OUT = {"config": {"sizes": SIZES, "seq": SEQ, "n_fit": N_FIT, "n_eval": N_EVAL, "n_concepts": N_CONC}, "models": {}}
T0 = time.time()


def save():
    json.dump(OUT, open("a1.json", "w"), indent=1)


def tokens():
    if TEST:
        return np.random.default_rng(0).integers(0, 300, size=200000)
    r = requests.get(DATA, headers={"Range": f"bytes=0-{2049 * 2 * 3000 - 1}"}, timeout=300)
    return np.frombuffer(r.content, dtype=np.uint16).reshape(-1, 2049)[:, :2048].reshape(-1)


def load(name, random_init=False):
    if TEST:
        d = 32 if name == "t-a" else 48
        torch.manual_seed(0 if not random_init else 9)
        return GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=300, hidden_size=d, num_hidden_layers=4, num_attention_heads=4,
                                                intermediate_size=4 * d, max_position_embeddings=256)).eval()
    if random_init:
        cfg = GPTNeoXConfig.from_pretrained("EleutherAI/" + name)
        torch.manual_seed(9)
        return GPTNeoXForCausalLM(cfg).eval()
    return AutoModelForCausalLM.from_pretrained("EleutherAI/" + name, dtype=torch.float32).eval()


def concepts(toks, tok):
    if TEST:
        return list(range(10, 10 + N_CONC))
    counts = np.bincount(toks[:4_000_000].astype(np.int64), minlength=len(tok))
    cands = [int(i) for i in np.argsort(-counts) if i < len(tok) and counts[i] > 0
             and re.fullmatch(r" [a-z]{4,}", tok.decode([int(i)]) or "")]
    words = [t[0] for w in WORDS for t in [tok(" " + w, add_special_tokens=False)["input_ids"]] if len(t) == 1]
    out = sorted(set(words))
    for c in cands[40:]:
        if len(out) >= N_CONC:
            break
        if c not in out:
            out.append(c)
    return sorted(out[:N_CONC])


def hidden(model, ids):
    with torch.no_grad():
        return model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states   # emb, L1..L


def fit_jlens(model, fit, cids):
    """U_l (concepts x d) for every layer l < L: J_l^T w_c averaged over source positions (sum over later positions)."""
    layers = model.gpt_neox.layers; L = len(layers)
    W = model.get_output_embeddings().weight.float()
    acts = {}
    hooks = [layers[i].register_forward_hook(lambda m, a, o, i=i: acts.__setitem__(i, o[0] if isinstance(o, tuple) else o))
             for i in range(L)]
    hooks.append(model.get_input_embeddings().register_forward_hook(lambda m, a, o: o.detach().requires_grad_(True)))
    valid = torch.zeros(SEQ, dtype=torch.bool); valid[8:SEQ - 1] = True
    U = torch.zeros(L - 1, len(cids), W.shape[1])
    chunk = CHUNK if W.shape[1] < 1024 else max(4, CHUNK // 8)      # 410m ran out of memory at 64 (run 45292404797)
    for ids in fit:
        for b in range(0, len(cids), chunk):
            ch = cids[b:b + chunk]; acts.clear()
            model(input_ids=ids[None].repeat(len(ch), 1), use_cache=False)
            cot = torch.zeros_like(acts[L - 1]); cot[:, valid] = W[ch][:, None, :]
            gs = torch.autograd.grad(acts[L - 1], [acts[i] for i in range(L - 1)], grad_outputs=cot, allow_unused=True)
            for i, g in enumerate(gs):
                if g is not None:
                    U[i, b:b + len(ch)] += g[:, valid].mean(1).detach()
    for h in hooks:
        h.remove()
    return U / len(fit)


def labels_future(ev, cids):
    c2i = {c: i for i, c in enumerate(cids)}
    lab = np.zeros((ev.shape[0], SEQ, len(cids)), dtype=bool)
    e = ev.numpy()
    for s in range(e.shape[0]):
        for t in range(SEQ):
            for u in e[s, t + 1:t + 1 + H]:
                j = c2i.get(int(u))
                if j is not None:
                    lab[s, t, j] = True
    return lab[:, 8:SEQ - H].reshape(-1, len(cids))


def mean_auc(S, Y):
    out = []
    for j in range(Y.shape[1]):
        y = Y[:, j]; n1 = y.sum(); n0 = len(y) - n1
        if n1 < 3 or n0 < 3:
            continue
        r = np.argsort(np.argsort(S[:, j])) + 1
        out.append((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(out)) if out else None


def pr(evals):
    return float(evals.sum() ** 2 / (evals ** 2).sum())


def main():
    toks = tokens()
    tok = None if TEST else AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    cids = concepts(toks, tok)
    seqs = torch.tensor(toks[: (N_FIT + N_EVAL) * SEQ].astype(np.int64)).view(-1, SEQ)
    fit, ev = seqs[:N_FIT], seqs[N_FIT:N_FIT + N_EVAL]
    Y = labels_future(ev, cids)
    OUT["config"]["concepts_used"] = len(cids); OUT["config"]["label_rate"] = float(Y.mean())
    store = {}
    for name in SIZES:
        t0 = time.time()
        m = load(name); L = m.config.num_hidden_layers; d = m.config.hidden_size; nh = m.config.num_attention_heads
        lstar = max(1, round(L / 3))
        U = fit_jlens(m, fit, cids)                                     # (L-1, C, d)
        hs = hidden(m, ev)
        rec = {"layers": L, "width": d, "heads": nh, "lstar": lstar, "per_layer": []}
        lnf, Wout = m.gpt_neox.final_layer_norm, m.get_output_embeddings().weight
        for l in range(1, L):
            h = hs[l][:, 8:SEQ - H].reshape(-1, d)
            with torch.no_grad():
                SL = (lnf(h) @ Wout[cids].T).numpy()
            SJ = (h @ U[l - 1].T).numpy()
            rec["per_layer"].append({"layer": l, "auc_jlens": mean_auc(SJ, Y), "auc_logit": mean_auc(SL, Y)})
        # (1)+(2) board at l*: shape, and content quality when only the top-k directions are kept
        Ul = U[lstar - 1]; Uc = Ul - Ul.mean(0, keepdim=True)
        _, sv, Vt = torch.linalg.svd(Uc, full_matrices=False)
        ev2 = (sv ** 2).numpy(); cum = np.cumsum(ev2) / ev2.sum()
        k90 = int((cum < 0.90).sum()) + 1
        h = hs[lstar][:, 8:SEQ - H].reshape(-1, d)
        q = {}
        for k in KS:
            if k > Vt.shape[0]:
                continue
            P = Vt[:k].T @ Vt[:k]
            q[str(k)] = mean_auc((h @ P @ Ul.T).numpy(), Y)
        rec["board"] = {"pr": pr(ev2), "k90": k90, "k90_over_width": k90 / d, "auc_full": rec["per_layer"][lstar - 1]["auc_jlens"],
                        "auc_logit": rec["per_layer"][lstar - 1]["auc_logit"], "auc_topk": q}
        Vs = Vt[:k90]                                                     # board subspace (k90 x d)
        base = k90 / d
        # (3) writers: share of each head / MLP output (layers before l*) that lands in the board subspace
        wr = []
        for li in range(lstar):
            blk = m.gpt_neox.layers[li]; got = {}
            h1 = blk.attention.dense.register_forward_pre_hook(lambda mod, a: got.__setitem__("att", a[0].detach()))
            h2 = blk.mlp.register_forward_hook(lambda mod, a, o: got.__setitem__("mlp", o.detach()))
            with torch.no_grad():
                m(input_ids=ev[:8], use_cache=False)
            h1.remove(); h2.remove()
            Wd = blk.attention.dense.weight.detach(); dh = d // nh
            for hi in range(nh):
                o = got["att"][..., hi * dh:(hi + 1) * dh].reshape(-1, dh) @ Wd[:, hi * dh:(hi + 1) * dh].T
                wr.append({"layer": li + 1, "part": f"head{hi}", "share": float((o @ Vs.T).pow(2).sum() / (o.pow(2).sum() + 1e-9))})
            o = got["mlp"].reshape(-1, d)
            wr.append({"layer": li + 1, "part": "mlp", "share": float((o @ Vs.T).pow(2).sum() / (o.pow(2).sum() + 1e-9))})
        # (4) readers: share of each later head's q/k/v and each MLP's input weights that reads the board subspace
        rd = []
        for li in range(lstar, L):
            blk = m.gpt_neox.layers[li]
            W = blk.attention.query_key_value.weight.detach() * blk.input_layernorm.weight.detach()[None]
            Wh = W.view(nh, 3, d // nh, d)
            for hi in range(nh):
                for qi, qn in enumerate("qkv"):
                    w = Wh[hi, qi]
                    rd.append({"layer": li + 1, "part": f"head{hi}.{qn}", "share": float((w @ Vs.T).pow(2).sum() / w.pow(2).sum())})
            w = blk.mlp.dense_h_to_4h.weight.detach() * blk.post_attention_layernorm.weight.detach()[None]
            rd.append({"layer": li + 1, "part": "mlp", "share": float((w @ Vs.T).pow(2).sum() / w.pow(2).sum())})
        rec["writers"] = {"random_baseline": base, "n": len(wr), "n_over_3x": sum(r["share"] > 3 * base for r in wr),
                          "top": sorted(wr, key=lambda r: -r["share"])[:8]}
        rec["readers"] = {"random_baseline": base, "n": len(rd), "n_over_3x": sum(r["share"] > 3 * base for r in rd),
                          "top": sorted(rd, key=lambda r: -r["share"])[:8]}
        rec["sec"] = round(time.time() - t0)
        G = torch.nn.functional.normalize(Uc, dim=1); store[name] = {"gram": (G @ G.T).numpy(), "h": h, "lstar": lstar}
        OUT["models"][name] = rec; save()
        print(name, "board", rec["board"], "writers>3x", rec["writers"]["n_over_3x"], "/", rec["writers"]["n"],
              "readers>3x", rec["readers"]["n_over_3x"], "/", rec["readers"]["n"], f"({time.time() - T0:.0f}s)", flush=True)
        if name == BIG:
            store[name]["target"] = (h @ Ul.T).numpy()
        del m
    # (5) similarity of concept geometry to the biggest model's board (off-diagonal correlation of the Gram matrices)
    iu = np.triu_indices(len(cids), 1); gb = store[BIG]["gram"][iu]
    OUT["geometry_vs_big"] = {n: float(np.corrcoef(store[n]["gram"][iu], gb)[0, 1]) for n in SIZES}
    # (6) does a smaller model's own state already hold the big board's content? ridge R2, with baselines
    T = store[BIG]["target"]; n = T.shape[0]; cut = int(n * 0.7)
    def r2(X):
        X = X.numpy() if torch.is_tensor(X) else X
        X = np.concatenate([X, np.ones((n, 1))], 1); best = -9
        for a in (1e-2, 1, 1e2, 1e4):
            A = X[:cut]; w = np.linalg.solve(A.T @ A + a * np.eye(A.shape[1]), A.T @ T[:cut])
            p = X[cut:] @ w
            best = max(best, 1 - ((T[cut:] - p) ** 2).sum() / ((T[cut:] - T[cut:].mean(0)) ** 2).sum())
        return float(best)
    rr = {}
    for name in SIZES:
        if name == BIG:
            continue
        m = load(name)
        hs = hidden(m, ev)
        rr[name] = {"own_state_at_lstar": r2(hs[store[name]["lstar"]][:, 8:SEQ - H].reshape(n, -1)),
                    "token_identity_only": r2(hs[0][:, 8:SEQ - H].reshape(n, -1))}
        mr = load(name, random_init=True)
        rr[name]["random_init_model"] = r2(hidden(mr, ev)[store[name]["lstar"]][:, 8:SEQ - H].reshape(n, -1))
        OUT["big_content_from_small"] = rr; save()
        print("ridge", name, rr[name], flush=True)
        del m, mr
    OUT["sec"] = round(time.time() - T0); save()
    print("GEOMETRY", json.dumps(OUT["geometry_vs_big"]), flush=True)
    for n_, r_ in OUT["models"].items():
        print("PERLAYER", n_, json.dumps([[x["layer"], round(x["auc_jlens"] or 0, 4), round(x["auc_logit"] or 0, 4)] for x in r_["per_layer"]]), flush=True)


if __name__ == "__main__":
    main()
