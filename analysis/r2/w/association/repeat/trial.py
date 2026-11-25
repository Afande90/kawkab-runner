"""First controlled lesson. Freeze the targets, prompts and prediction before running."""
import torch,json
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path(__file__).resolve().parents[2]
torch.set_num_threads(6)
tok=AutoTokenizer.from_pretrained(root/'jinstall/donor')
d=torch.load(root/'capacity/layer5-map.pt')['transport']
colors=[' green',' black']
ids=[tok(s).input_ids for s in colors]
assert all(len(i)==1 for i in ids)
# Probe after tag before any color appears. Training and testing use different wording.
tags=['Tovin','Neral']
templates=['The paint assigned to {} is','The recorded color for {} is','They marked {} with the color','In the table, {} has color']
probes=['The new badge for {} should be painted','The official shade assigned to {} is']
def tokens(s):return tok(s,return_tensors='pt').input_ids
out={}
for arm in [0,1]:
 m=AutoModelForCausalLM.from_pretrained(root/'jinstall/donor',torch_dtype=torch.float32).eval()
 w=m.embed_out.weight.detach()
 lens=(w[ids[0][0]]-w[ids[1][0]])@d
 z=[]
 def hook(mod,args,o):z.append(o[0][0,-1].detach())
 h=m.gpt_neox.layers[5].register_forward_hook(hook)
 @torch.no_grad()
 def probe():
  rows=[]
  for tag in tags:
   for q in probes:
    l=m(tokens(q.format(tag)),use_cache=False).logits[0,-1]
    rows.append([float(z.pop()@lens),float(l[ids[0][0]]-l[ids[1][0]])])
  return rows
 base=probe()
 # Identical intervention in the early layers in both arms. Freeze remaining weights.
 m.requires_grad_(False)
 params=list(m.gpt_neox.layers[5].parameters())
 for p in params:p.requires_grad_(True)
 opt=torch.optim.AdamW(params,lr=0.0001,weight_decay=0)
 for step in range(12):
  opt.zero_grad()
  for j,tag in enumerate(tags):
   l=m(tokens(templates[step%4].format(tag)),use_cache=False).logits[:,-1]
   loss=torch.nn.functional.cross_entropy(l,torch.tensor(ids[j^arm]))/2
   loss.backward();z.clear()
  opt.step()
 out[str(arm)]=dict(before=base,after=probe())
 h.remove();del m,opt,params,w
(root/'association/repeat/result.json').write_text(json.dumps(out,indent=2))
for metric in [0,1]:
 for j in range(2):
  x=[sum((r['after'][i][metric]-r['before'][i][metric]) for i in range(2*j,2*j+2))/2 for r in out.values()]
  print(metric,tags[j],x,'green-assignment minus black-assignment',(x[0]-x[1])*(1-2*j))
