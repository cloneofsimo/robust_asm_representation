import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
X = torch.load("tok_emb.dat").cpu().numpy()[:400, :]

print(X.shape)
Xd = TSNE(n_components=2, n_iter = 8000, perplexity= 600).fit_transform(X)


fig, ax = plt.subplots()
x, y = Xd[:, 0].tolist(), Xd[:, 1].tolist()

ax.scatter(x, y)
#plt.show()

import json
tok_idx = json.load(open("./toks.json"))
tok_key = [k for v, k in tok_idx]
for i, txt in enumerate(tok_key[:400]):
    ax.annotate(txt, (x[i], y[i]))

plt.show()