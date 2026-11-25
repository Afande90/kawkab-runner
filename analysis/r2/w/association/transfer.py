"""First controlled lesson. Freeze the targets, prompts and prediction before running."""
import torch,json
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
root=Path(__file__).resolve().parent.parent
torch.set_num_threads(6)
tok=AutoTokenizer.from_pretrained(root/'jinstall/donor')
d=torch.load(root/'capacity/layer5-map.pt')['transport']
colors=[' red',' blue']
ids=[tok(s).input_ids for s in colors]
assert all(len(i)==1 for i in ids)
# Probe after tag before any color appears. Training and testing use different wording.
tags=['Kelm','Vorn']
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
 # Test the observed training change in the earlier partial-graft recipient.
 import sys
 sys.path.insert(0,str(root/'intact'))
 from load_recipient import load
 from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel
 from transformers import GPTNeoXConfig
 from torch import nn
 pre,sm,_=load(root/'intact/installed',256)
 sm.bridge.weight.data.copy_(torch.load(root/'capacity/layer5-map.pt')['axes'].T)
 @torch.no_grad()
 def assess(prefix):
  scores=[]
  for j,tag in enumerate(tags):
   for q in probes:
    l=sm(prefix,tokens(q.format(tag)))[1][0,-1]
    scores.append(float(l[ids[j^arm][0]]-l[ids[1^(j^arm)][0]]))
  return scores
 prior=assess(pre)
 pre.layers[5].load_state_dict(m.gpt_neox.layers[5].state_dict())
 after=assess(pre)
 del pre
 pre,sm,_=load(root/'intact/installed',256)
 del pre
 torch.manual_seed(7101)
 cfg=GPTNeoXConfig.from_json_file(str(root/'intact/installed/donor-config.json'));cfg.num_hidden_layers=6
 normal=GPTNeoXModel(cfg).eval();normal.final_layer_norm=nn.Identity()
 no_copy=assess(normal)
 out[str(arm)]['small']={'before':prior,'after':after,'no_copy':no_copy}
 del normal,sm
 h.remove();del m,opt,params,w
(root/'association/transfer.json').write_text(json.dumps(out,indent=2))
for metric in [0,1]:
 for j in range(2):
  x=[sum((r['after'][i][metric]-r['before'][i][metric]) for i in range(2*j,2*j+2))/2 for r in out.values()]
  print(metric,tags[j],x,'red-assignment minus blue-assignment',(x[0]-x[1])*(1-2*j))
