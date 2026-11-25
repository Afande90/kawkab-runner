"""Load a packaged recipient and predict a next token without the original donor."""
from pathlib import Path
import argparse
import torch
from torch import nn
from transformers import GPTNeoXConfig, AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXModel
from safetensors.torch import load_file
from transplant import Recipient

def load(root,width):
    c=GPTNeoXConfig.from_json_file(str(root/'donor-config.json'))
    depth={128:4,256:8}[width]
    recipient=Recipient(c,width,depth).eval()
    recipient.load_state_dict(load_file(str(root/f'recipient-{width}-{depth}.safetensors')))
    c.num_hidden_layers=6
    prefix=GPTNeoXModel(c).eval()
    prefix.final_layer_norm=nn.Identity()
    prefix.load_state_dict(load_file(str(root/'donor-prefix.safetensors')))
    return prefix,recipient,AutoTokenizer.from_pretrained(root)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--width',type=int,choices=[128,256],default=128)
    p.add_argument('--prompt',default='The lights went out when')
    args=p.parse_args()
    torch.set_num_threads(6)
    prefix,recipient,tok=load(Path(__file__).parent/'installed',args.width)
    with torch.inference_mode():
        _,scores=recipient(prefix,tok(args.prompt,return_tensors='pt').input_ids)
        print(repr(tok.decode([scores[0,-1].argmax().item()])))
