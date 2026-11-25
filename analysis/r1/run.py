# r1 — fresh reproduction of Phase 1 (Pythia-410m, steps 2000 and 33000) with the ORIGINAL, unmodified Phase 1 code
# (phase1_original.py = kawkab-private commit eae294b kernels/phase1/run.py, checked by sha256 below).
# Only the model list and step list are narrowed; texts, concept list, settings and metric code are the original's.
import hashlib, json, platform, sys, time
import torch, transformers, datasets, numpy as np
from huggingface_hub import HfApi

EXPECTED = "5432432adf0de3029b94890184c6d9a71003f969849a70cb7d36d0d9f1d51a59"
src = open("phase1_original.py", "rb").read()
assert hashlib.sha256(src).hexdigest() == EXPECTED, "original script changed"
import phase1_original as P

NAME, STEPS = "EleutherAI/pythia-410m", [2000, 33000]
api = HfApi()
out = {"kind": "FRESH RUN", "script_sha256": EXPECTED, "device": P.dev,
       "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                    "datasets": datasets.__version__, "numpy": np.__version__},
       "settings": {k: getattr(P, k) for k in ("SEQ", "SKIP", "BATCH", "N_FIT", "N_EVAL", "AHEAD")},
       "dataset": {"id": "Salesforce/wikitext", "config": "wikitext-103-raw-v1", "split": "test",
                   "sha": api.dataset_info("Salesforce/wikitext").sha},
       "revisions": {}, "results": {}}
tok = P.AutoTokenizer.from_pretrained(NAME)
fit, ev = P.texts(tok)
cids = P.concept_ids(tok)
out["n_concepts"] = len(cids)
out["text_hash"] = hashlib.sha256(json.dumps([fit, ev]).encode()).hexdigest()
out["concept_ids"] = cids
for step in STEPS:
    out["revisions"][f"step{step}"] = api.model_info(NAME, revision=f"step{step}").sha
    t0 = time.time()
    rows = P.run(NAME, step, fit, ev, cids)
    out["results"][str(step)] = rows
    r = rows[5]
    print(f"FRESH {NAME} step{step} rev {out['revisions'][f'step{step}']} layer {r['layer']} "
          f"auc_jlens {r['auc_jlens']:.6f} auc_logit {r['auc_logit']:.6f} pr {r['pr']:.2f} rank90 {r['rank90']} "
          f"({time.time() - t0:.0f}s)", flush=True)
    json.dump(out, open("r1.json", "w"))
print("META", json.dumps({k: out[k] for k in ("versions", "settings", "dataset", "revisions", "n_concepts", "text_hash", "device")}), flush=True)
print("CONCEPTS", json.dumps(cids), flush=True)
