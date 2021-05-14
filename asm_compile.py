  

import random
from subprocess import Popen, PIPE

from tqdm import tqdm
from glob import glob

import json


def __main():
    """
    Not to be exported.
    """
    probs = set(json.load(open("idx2u.json")))

    print("N_Problems", probs)

    pbar = tqdm(glob("./dataset/source/*/*.cpp"))
    for fpath in pbar:
        prob = fpath.split('\\')[-1][:-4]
        
        pbar.set_description(prob)
        if prob in probs:
            p = Popen(['g++', fpath, '-S', '-masm=intel', '-o', "./dataset/asms/" + prob + ".S"], shell=True, stdout=PIPE, stdin=PIPE)


if __name__ == "__main__":
    __main()