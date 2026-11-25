"""Intact donor-prefix graft into a fresh narrow GPT-NeoX recipient, no training."""
from pathlib import Path
from copy import deepcopy
import argparse, json
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM
from safetensors.torch import save_file, load_file

MODEL='EleutherAI/pythia-410m'
REV='962a951eac83aa52f9f013bcfb61853cc1d434d1'

class Recipient(nn.Module):
    def __init__(self,c,width,depth):
        super().__init__()
        upper=deepcopy(c)
        upper.hidden_size=width
        upper.intermediate_size=4*width
        upper.num_attention_heads=width//64
        upper.num_hidden_layers=depth
        self.bridge=nn.Linear(c.hidden_size,width,bias=False)
        self.upper=GPTNeoXForCausalLM(upper)
        self.upper.gpt_neox.embed_in=None  # Input comes from the graft.
    def forward(self,prefix,ids):
        z=prefix(input_ids=ids,use_cache=False).last_hidden_state
        logits=self.upper(inputs_embeds=self.bridge(z),use_cache=False).logits
        return z,logits

def prefix_from(donor):
    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel
    c=deepcopy(donor.config)
    c.num_hidden_layers=6
    core=GPTNeoXModel(c)
    core.final_layer_norm=nn.Identity()
    src=donor.gpt_neox.state_dict()
    core.load_state_dict({k:src[k] for k in core.state_dict()},strict=True)
    return core.eval()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--donor',default=MODEL)
    p.add_argument('--output',type=Path,default=Path('installed'))
    args=p.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(3117)
    path=args.output
    path.mkdir(parents=True,exist_ok=True)
    donor=AutoModelForCausalLM.from_pretrained(args.donor,revision=REV,torch_dtype=torch.float32).eval()
    tok=AutoTokenizer.from_pretrained(args.donor,revision=REV)
    prefix=prefix_from(donor)
    # Save the standalone prefix at donor precision; both recipients use it.
    save_file(prefix.state_dict(),str(path/'donor-prefix.safetensors'))
    donor.config.to_json_file(str(path/'donor-config.json'))
    tok.save_pretrained(path)
    pairs=[l.split('\t') for l in (Path(__file__).parent/'prompts.tsv').read_text().splitlines()]
    refs=[]
    state=[]
    h=donor.gpt_neox.layers[5].register_forward_hook(lambda m,a,out: state.append(out[0].clone()))
    for s,a,t,b in pairs:
        for prompt,right,alt in ((s,a,b),(t,b,a)):
            ids=tok(prompt,return_tensors='pt').input_ids
            own,other=[tok(' '+w).input_ids for w in (right,alt)]
            assert len(own)==len(other)==1
            logits=donor(ids,use_cache=False).logits[0,-1]
            refs.append((ids,own[0],other[0],state.pop(),float(logits[own[0]]-logits[other[0]])))
    h.remove()
    # Reload the actual saved graft before testing it in each recipient.
    prefix.load_state_dict(load_file(str(path/'donor-prefix.safetensors')))
    result={'prefix_parameters':sum(p.numel() for p in prefix.parameters()),
            'donor_parameters':sum(p.numel() for p in donor.parameters()),
            'donor_correct':sum(r[-1]>0 for r in refs),'prompts':len(refs),'recipients':{}}
    for width,depth in [(128,4),(256,8)]:
        student=Recipient(donor.config,width,depth).eval()
        weights=path/f'recipient-{width}-{depth}.safetensors'
        save_file(student.state_dict(),str(weights))
        student.load_state_dict(load_file(str(weights)))
        errors=[]; margins=[]
        for ids,own,other,target,_ in refs:
            z,logits=student(prefix,ids)
            errors.append(float((z-target).abs().max()))
            margins.append(float(logits[0,-1,own]-logits[0,-1,other]))
        result['recipients'][str(width)]=dict(depth=depth,parameters=result['prefix_parameters']+sum(p.numel() for p in student.parameters()),max_state_error=max(errors),correct=sum(m>0 for m in margins),margins=margins)
    (path/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result)[:450])

if __name__=='__main__': main()
