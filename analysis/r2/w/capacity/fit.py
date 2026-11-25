from pathlib import Path
import json,torch,time
from transformers import AutoModelForCausalLM,AutoTokenizer
root=Path(__file__).resolve().parent
d=root.parent/'jinstall/donor'
torch.set_num_threads(7)
model=AutoModelForCausalLM.from_pretrained(d,torch_dtype=torch.float32,attn_implementation='eager').eval().requires_grad_(False)
tok=AutoTokenizer.from_pretrained(d)
texts=[
'After a week of cloudy weather, the sun broke through and warmed the stones along the garden path.',
'Each afternoon, the clerk placed the stamped envelopes into a canvas sack before the mail train arrived.',
'Three students built a small bridge from sticks and measured how much weight the structure could hold.',
'A winding trail led from the coast through a pine forest to the edge of a deep lake.',
'The violinist stopped halfway through the rehearsal to tighten a string and mark the score with a pencil.',
'The bus driver checked the timetable while passengers waited beneath the old station clock in the rain.']
# Estimate the transport at every residual boundary for all output coordinates.
acts=[]
def at_input(mod,args,out):return out.detach().requires_grad_(True)
def keep(mod,args,out):acts.append(out[0])
model.gpt_neox.embed_in.register_forward_hook(at_input)
for block in model.gpt_neox.layers:block.register_forward_hook(keep)
J=torch.zeros(23,1024,1024)
eye=torch.eye(1024)
for n,s in enumerate(texts):
    acts.clear()
    model.gpt_neox(tok(s,return_tensors='pt').input_ids,use_cache=False)
    pooled=acts[-1][0,8:-1].sum(0)
    for b in range(0,1024,16):
        grads=torch.autograd.grad(pooled,acts[:-1],eye[b:b+16],is_grads_batched=True,retain_graph=b+16<1024)
        for i,g in enumerate(grads):J[i,b:b+16]+=g[:,0,8:-1].mean(1)
    print(n+1,flush=True)
J/=len(texts)
# Vocabulary is the tokenizer's IDs, excluding padded rows in the output matrix.
W=model.embed_out.weight[:len(tok)].double()
W=W-W.mean(0)
G=W.T@W
# Validate compressed scores on 64 existing prompts, unused for estimating J.
pairs=[s.split('\t') for s in (root.parent/'jinstall/input/prompts.tsv').read_text().splitlines()]
H=[]
with torch.no_grad():
    for a,_,b,_ in pairs:
        for s in (a,b):
            acts.clear()
            model.gpt_neox(tok(s,return_tensors='pt').input_ids,use_cache=False)
            H.append(torch.stack([x[0,-1].detach() for x in acts[:-1]]))
H=torch.stack(H).double()
rows=[]; maps={}
for i,A in enumerate(J.double()):
    eig,V=torch.linalg.eigh(A.T@G@A)
    eig=eig.flip(0).clamp_min(0); V=V.flip(1)
    total=eig.sum()
    h=H[:,i]
    activity=(h@V).square().sum(0)*eig
    row={'layer':i,'rank90':int((eig.cumsum(0)<.9*total).sum())+1}
    for r in [32,128,256,512]:
        row[str(r)]={'matrix_energy':float(eig[:r].sum()/total),'prompt_score_energy':float(activity[:r].sum()/activity.sum())}
    rows.append(row)
    if i==5:
        maps={'transport':A.float(),'axes':V[:,:256].float(),'squared_singular_values':eig.float()}
torch.save(maps,root/'layer5-map.pt')
(root/'measurements.json').write_text(json.dumps({'fit_texts':texts,'tokens':len(tok),'heldout':len(H),'rows':rows},indent=2))
print(rows[5],flush=True)
