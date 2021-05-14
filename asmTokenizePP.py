import random

from tqdm import tqdm
from glob import glob


asms = glob("./dataset/asms/*.S")

pbar = tqdm(asms)


import json

tok_idx = json.load(open("./toks.json"))
tok_idx = {val : idx for idx, (_, val) in enumerate(tok_idx)}

#print(tok_idx)

for asmfp in pbar:
    asm = str(open(asmfp, "r").read())
    prob = asmfp.split('\\')[-1][:-2]
        
    pbar.set_description(str(len(asm)))

    asm_toks = asm.replace('\t', ' ').replace('\n', ' ').split(' ')

    asm_tok_lis = []
    
    for tok in asm_toks:
        if tok == '':
            continue
        pbar.set_description(tok)
        if tok_idx.get(tok,0) == 0:
            continue
        else:
            asm_tok_lis.append(tok_idx[tok])

    with open("./dataset/tokenized/" + prob + ".json", 'w') as F:
        print(prob)
        json.dump(asm_tok_lis, F)
    

    