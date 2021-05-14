import random

from tqdm import tqdm
from glob import glob


asms = glob("./dataset/asms/*.S")[:500]

pbar = tqdm(asms)


tok_pq = {}

for asm in pbar:
    asm = str(open(asm, "r").read())
    
    pbar.set_description(str(len(asm)))

    asm_toks = asm.replace('\t', ' ').replace('\n', ' ').split(' ')
    
    for tok in asm_toks:
        if tok == '':
            continue
        pbar.set_description(tok)
        if tok_pq.get(tok,0) == 0:
            tok_pq[tok] = 1
        else:
            tok_pq[tok] += 1

pq = sorted([(v, k) for k, v in tok_pq.items()], reverse= True)

import json

json.dump(pq[:5000], open("toks.json", 'w'))
