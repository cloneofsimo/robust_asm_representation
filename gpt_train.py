import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
from tqdm import tqdm
from glob import glob
import math

from model_gpt import GPT
import random
import json

def top_k_logits(logits, k):
    v, _ = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out
    
@torch.no_grad()
def sample(model, x, steps, temperature=1.0, top_k=12):
    block_size = model.get_block_size()
    model.eval()
    for _ in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:] # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        
        logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        ix = torch.multinomial(probs, num_samples=1)
        x = torch.cat((x, ix), dim=1)
    return x

class AsmTok(Dataset):
    def __init__(self, block_size = 128):

        dpath = glob("./dataset/tokenized/*.json")

        self.dpath = []
        for fp in dpath:
            with open(fp) as F:
                val = len(json.load(F))
                if val > 200:
                    self.dpath.append(fp)
              

        self.block_size = block_size

        print("DATASET SIZE", len(self.dpath))

    def __len__(self):
        return len(self.dpath)

    def __getitem__(self, idx):

        val = self.dpath[idx]
        #print(val)
        data = json.load(open(self.dpath[idx], 'r'))
        #print(len(data))
        idx = random.randint(0, len(data) - self.block_size - 10)
        dix = data[idx:idx + self.block_size]

        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y



epochs = 60

batch_size = 64
lr = 1e-3
is_train = True

train_dataset = AsmTok()
device = torch.device("cuda:0")

model = GPT(vocab_size = 5000)
optimizer = torch.optim.AdamW(model.parameters(), lr = lr, weight_decay = 1e-9)
dl = DataLoader(train_dataset, shuffle=True, pin_memory= True, batch_size=batch_size)

model.to(device)
model.train()
if is_train:
    for epoch in range(epochs):
        pbar = tqdm(dl)
        for (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f"epoch {epoch+1} train loss {loss.item():.5f}")
    torch.save(model.state_dict(), "model.dat")
else:
    model.load_state_dict(torch.load("model.dat"))

torch.save(model.tok_emb.weight.data, "tok_emb.dat")
x = torch.tensor([[172]]).cuda()
y = sample(model, x, 64, temperature=1.0, top_k= 12)[0]

print(y)
"""
tensor([ 172,  170,  162,  107,   25,  116,  552,  117,   60,   48,    7,  678,
         113, 1183,   25,   27,   26,   48,   45,   29,   22,   20,   23,   19,
          18,   21,   34,   14,   16,   38,   12,   17,   36,   39,   32,   31,
          37,    6,    8,   10,  120,    2,    1,   86,   43,    3,    2,    1,
          71,   41,    9,   81,    2,    1,   11,   24,   13,    4,    5,    8,
          10,    7,    3,    2,    1]
"""