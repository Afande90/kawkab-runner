"""Phase 1 — REA the birth of J-space.

Fit a Jacobian lens (concept-token J-lens vectors u_l(t) = J_l^T w_t, J_l = E[d h_final / d h_l] summed over the
current and later positions, averaged over source positions and prompts) on public training checkpoints of small
models, and track how the workspace forms over training.

Per model, checkpoint and layer:
  rank90      dims holding 90% of the energy of the concept J-lens matrix (workspace width)
  pr          participation ratio of its singular values (effective dimensionality)
  norm        mean J-lens vector norm (how strongly the layer pushes towards concepts)
  auc_jlens   do the J-lens scores u_l(t).h_l at a position rank the concepts that actually appear in the next
              16 tokens above the ones that don't? (does the layer hold what is coming = workspace content)
  auc_logit   the same with the plain unembedding w_t.h_l (logit lens) — the baseline J-lens must beat
"""
import json
import time

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = ["EleutherAI/pythia-160m", "EleutherAI/pythia-410m"]
STEPS = [0, 128, 512, 1000, 2000, 4000, 8000, 16000, 33000, 66000, 100000, 143000]
SEQ, SKIP, BATCH, N_FIT, N_EVAL, AHEAD = 64, 8, 32, 48, 32, 16
WORDS = ("time day year man woman child water food house city country war king money law school family music "
         "body death life world light fire tree river book game team door road car ship church army power heart "
         "mother father son daughter friend doctor police court market price company government president party "
         "black white red green blue red cold hot fast slow good bad new old big small first last north south "
         "love fear hope pain help work play run walk speak write read sleep eat drink kill build open close "
         "number science history art language word name story question answer problem reason idea mind god "
         "village island mountain forest sea sun moon star night morning winter summer horse dog bird fish").split()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False


def texts(tok):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    chunks, buf = [], []
    for line in ds["text"]:
        if len(line.strip()) < 200:
            continue
        ids = tok(line, add_special_tokens=False)["input_ids"]
        if len(ids) >= SEQ:
            chunks.append(ids[:SEQ])
        if len(chunks) >= N_FIT + N_EVAL:
            break
    return chunks[:N_FIT], chunks[N_FIT:]


def concept_ids(tok):
    ids = []
    for w in WORDS:
        t = tok(" " + w, add_special_tokens=False)["input_ids"]
        if len(t) == 1:
            ids.append(t[0])
    return sorted(set(ids))


def find_layers(model):
    n = model.config.num_hidden_layers
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) == n:
            return m
    raise RuntimeError("no layer list")


def auc(scores, labels):
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def run(name, step, fit, ev, cids):
    model = AutoModelForCausalLM.from_pretrained(name, revision=f"step{step}", torch_dtype=torch.float32).to(dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    L = len(layers)
    W = model.get_output_embeddings().weight  # [vocab, d]
    acts = {}

    def keep(i):
        def hook(_m, _i, out):
            acts[i] = out[0] if isinstance(out, tuple) else out
        return hook

    hooks = [layer.register_forward_hook(keep(i)) for i, layer in enumerate(layers)]
    emb = model.get_input_embeddings()
    hooks.append(emb.register_forward_hook(lambda _m, _i, out: out.detach().requires_grad_(True)))
    d = W.shape[1]
    U = torch.zeros(L - 1, len(cids), d, device=dev)
    valid = torch.zeros(SEQ, dtype=torch.bool, device=dev)
    valid[SKIP:SEQ - 1] = True
    for ids in fit:
        for b in range(0, len(cids), BATCH):
            chunk = cids[b:b + BATCH]
            acts.clear()
            model(input_ids=torch.tensor([ids] * len(chunk), device=dev), use_cache=False)
            final = acts[L - 1]
            cot = torch.zeros_like(final)
            cot[:, valid, :] = W[chunk][:, None, :]
            grads = torch.autograd.grad(final, [acts[i] for i in range(L - 1)], grad_outputs=cot, allow_unused=True)
            for li, g in enumerate(grads):
                if g is not None:
                    U[li, b:b + len(chunk)] += g[:, valid, :].mean(1)
    U /= len(fit)
    for h in hooks[-1:]:
        h.remove()
    # evaluation: does the layer hold the concepts that are coming?
    cset = torch.tensor(cids, device=dev)
    rows = []
    with torch.no_grad():
        hs = {i: [] for i in range(L - 1)}
        labels = []
        for ids in ev:
            acts.clear()
            model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
            t = torch.tensor(ids, device=dev)
            for p in range(SKIP, SEQ - AHEAD):
                fut = t[p + 1:p + 1 + AHEAD]
                labels.append(torch.isin(cset, fut).cpu().numpy())
                for i in range(L - 1):
                    hs[i].append(acts[i][0, p])
        labels = np.stack(labels)
        Wc = W[cids]
        for li in range(L - 1):
            H = torch.stack(hs[li])  # [n, d]
            m = U[li]
            s = torch.linalg.svdvals(m)
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            sj = (H @ m.T).cpu().numpy()
            sl = (H @ Wc.T).cpu().numpy()
            # rank-normalise per position so the AUC measures which concepts, not how loud the vector is
            sj = sj - sj.mean(1, keepdims=True)
            sl = sl - sl.mean(1, keepdims=True)
            rows.append({"layer": li, "rank90": int((e < 0.9).sum()) + 1,
                         "pr": float((s ** 2).sum() ** 2 / (s ** 4).sum()),
                         "norm": float(m.norm(dim=1).mean()),
                         "auc_jlens": auc(sj.ravel(), labels.ravel()),
                         "auc_logit": auc(sl.ravel(), labels.ravel())})
    for h in hooks:
        h.remove()
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    out = {"words": WORDS, "steps": STEPS, "results": {}}
    for name in MODELS:
        tok = AutoTokenizer.from_pretrained(name)
        fit, ev = texts(tok)
        cids = concept_ids(tok)
        out["results"][name] = {}
        for step in STEPS:
            t0 = time.time()
            try:
                out["results"][name][step] = run(name, step, fit, ev, cids)
            except Exception as ex:  # keep going; record the failure
                out["results"][name][step] = {"error": str(ex)[:300]}
            mid = out["results"][name][step]
            if isinstance(mid, list):
                m = mid[len(mid) // 2]
                print(f"{name} step {step}: concepts {len(cids)} mid-layer rank90 {m['rank90']} pr {m['pr']:.1f} "
                      f"auc_j {m['auc_jlens']} auc_logit {m['auc_logit']} ({time.time() - t0:.0f}s)", flush=True)
            json.dump(out, open("phase1.json", "w"))


if __name__ == "__main__":
    main()
