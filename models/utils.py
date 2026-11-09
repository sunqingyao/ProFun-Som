from argparse import ArgumentParser
import os

import torch

from Bio.SeqIO import parse
from Bio.SeqRecord import SeqRecord
from typing import Generator

def to_np(x):
    return x.cpu().detach().numpy()

def parsing(parser : ArgumentParser):
    opt = parser.parse_args()

    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt

def record_generator(fasta_path: str) -> Generator[SeqRecord, None, None]:
    with open(fasta_path, "r") as h:
        for record in parse(h, "fasta"):
            yield record