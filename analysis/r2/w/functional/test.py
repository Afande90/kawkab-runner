from pathlib import Path
import sys,json,csv
import torch
from torch import nn
from transformers import AutoModelForCausalLM,GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'intact'))
from load_recipient import load
torch.set_num_threads(6);torch.set_grad_enabled(False)
width=256
axes=torch.load(root/'capacity/layer5-map.pt')['axes'][:,:width]
prefix,student,tok=load(root/'intact/installed',width)
donor=AutoModelForCausalLM.from_pretrained(root/'jinstall/donor',torch_dtype=torch.float32).eval()
pairs=[x.split('\t') for x in (root/'functional/pairs.tsv').read_text().splitlines()]
records=[]
def test(name,run):
 for k,(s,a,t,b) in enumerate(pairs):
    right,alt=[tok(' '+w).input_ids for w in (a,b)]
    if len(right)!=1 or len(alt)!=1:
        raise ValueError((a,right,b,alt))
    zs,ls=run(s,None);zt,lt=run(t,None)
    for prompt,edit,i,j,baseline in [(s,zt-zs,right[0],alt[0],ls),(t,zs-zt,alt[0],right[0],lt)]:
        _,patched=run(prompt,edit)
        margin=float(baseline[i]-baseline[j]);post=float(patched[i]-patched[j])
        records.append(dict(model=name,pair=k+1,prompt=prompt,correct=margin>0,margin=margin,shift=margin-post))
def teacher(s,edit):
 state=[]
 def hook(m,args,out):
    z=out[0];state.append(z[0,-1]@axes)
    if edit is not None:
        z=z.clone();z[0,-1]+=axes@edit
        return (z,)+out[1:]
 h=donor.gpt_neox.layers[5].register_forward_hook(hook)
 try:l=donor(tok(s,return_tensors='pt').input_ids,use_cache=False).logits[0,-1]
 finally:h.remove()
 return state[0],l
test('donor',teacher)
del donor
def small(s,edit):
 ids=tok(s,return_tensors='pt').input_ids
 z=student.bridge(prefix(ids,use_cache=False).last_hidden_state)
 before=z[0,-1].clone()
 if edit is not None:z[0,-1]+=edit
 return before,student.upper(inputs_embeds=z,use_cache=False).logits[0,-1]
fresh_bridge=student.bridge.weight.clone()
student.bridge.weight.copy_(axes.T)
test('graft',small)
student.bridge.weight.copy_(fresh_bridge)
torch.manual_seed(7101)
c=GPTNeoXConfig.from_json_file(str(root/'intact/installed/donor-config.json'))
c.num_hidden_layers=6
prefix=GPTNeoXModel(c).eval();prefix.final_layer_norm=nn.Identity()
test('no_copy',small)
with (root/'functional/answers.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
for name in ['donor','graft','no_copy']:
 r=[x for x in records if x['model']==name]
 print(name,len(r),sum(x['correct'] for x in r),sum(x['shift'] for x in r)/len(r))
