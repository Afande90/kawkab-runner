#!/bin/bash
# r2: independent rebuild + rerun of ChatGPT's "JSpace-Predict-Then-Train" package from its own source pack.
# Missing inputs rebuilt here: jinstall/donor (= EleutherAI/pythia-410m rev 962a951e, as named in its transplant.py);
# jinstall/input/prompts.tsv and intact/prompts.tsv (only feed side measurements) = its functional/pairs.tsv.
set -e
cd w
python -c "
from transformers import AutoModelForCausalLM,AutoTokenizer
r='962a951eac83aa52f9f013bcfb61853cc1d434d1'
AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-410m',revision=r).save_pretrained('jinstall/donor')
AutoTokenizer.from_pretrained('EleutherAI/pythia-410m',revision=r).save_pretrained('jinstall/donor')"
mkdir -p jinstall/input && cp ../prompts_stub.tsv jinstall/input/prompts.tsv && cp ../prompts_stub.tsv intact/prompts.tsv
echo "== CHECK"; python audit/check.py
echo "== FIT"; python capacity/fit.py | tail -1
python -c "import torch,hashlib;d=torch.load('capacity/layer5-map.pt');print('MAPSUM transport',float(d['transport'].double().abs().sum()),'axes',float(d['axes'].double().abs().sum()))"
echo "== TRANSPLANT"; (cd intact && python transplant.py --donor ../jinstall/donor --output installed | tail -1)
for s in association/trial.py association/repeat/trial.py association/transfer.py association/repeat/transfer.py functional/test.py; do echo "== RUN $s"; python -u $s; done
python - <<'P'
import json
for n in ('association','association/repeat'):
    d=json.load(open(n+'/transfer.json'))
    for k in ['before','after','no_copy']:
        v=[x for a in d.values() for x in a['small'][k]]; print('SMALL',n,k,f'correct={sum(x>0 for x in v)}/{len(v)}',[round(x,4) for x in v])
P
