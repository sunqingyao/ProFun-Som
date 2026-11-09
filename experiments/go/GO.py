import pickle
from typing import Deque, Iterator, Match, Optional, Set, Tuple, List, Dict, Iterable
from typing import Generator
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn import functional as F
import numpy as np
import os
from Bio.SeqRecord import SeqRecord

import sys
prj_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if prj_path not in sys.path:
    sys.path.append(prj_path)

from experiments.preprocess.utils import reweight, fast_dca, msa2pssm
from experiments.preprocess.alignparser import (parsing_a3m_by_range, 
                                                seqrecord2numpy, 
                                                encoding, 
                                                select_from_generator_by_index)

class MSAEncoder(nn.Module):
    """
    """
    num_embeddings = 21
    def __init__(self, cutoff : float, penalty : float, embedding_dim : int, 
                 encoding_strategy : str = "pssm_plus"):
        super(MSAEncoder, self).__init__()
        self.cutoff = cutoff
        self.penalty = penalty
        self.embedding_dim = embedding_dim
        self.encoding_strategy = encoding_strategy
        self.emb_layer = nn.Embedding(num_embeddings=self.num_embeddings, 
                                  embedding_dim=self.embedding_dim)

        assert encoding_strategy in ["one_hot", "emb", 
                                     "emb_plus_one_hot", "emb_plus_pssm", 
                                     "fast_dca"],\
            f"the encoding strategy {encoding_strategy} is not implemented"
        if encoding_strategy in ["one_hot", "emb", "emb_plus_one_hot"]:
            self.enc_layer = None 
        elif encoding_strategy == "emb_plus_pssm":
            self.enc_layer = PSSMPlus(self.cutoff)  
        else:
            self.enc_layer = FastDCA(self.cutoff, self.penalty)
    
    def forward(self, input : Tensor):
        if self.encoding_strategy == "one_hot":
            x = F.one_hot(input, num_classes=self.embedding_dim)
        else:
            x = self.emb_layer(input)

        if self.enc_layer is not None:
            return self.enc_layer(x)
        elif self.encoding_strategy == "emb_plus_one_hot":
            return x + F.one_hot(input, num_classes=self.embedding_dim)
        else:
            return x.to(torch.float32) # long to float32

class PSSMPlus(nn.Module):
    def __init__(self, cutoff : float):
        super(PSSMPlus, self).__init__()
        self.cutoff = cutoff
    
    def forward(self, input : Tensor):
        """
        input: b x k x L x 21

        return : b x k x L x 21
        """
        weight = reweight(input, self.cutoff)
        pssm = msa2pssm(input, weight).unsqueeze(dim=-3) # b x 1 x L x 21
        return input + pssm

class FastDCA(nn.Module):
    def __init__(self, cutoff: float, penalty: float):
        super(FastDCA, self).__init__()
        self.cutoff = cutoff
        self.penalty = penalty
    
    def forward(self, input : Tensor):
        """
        input: b x k x L x 21

        return: b x L x L x 441
        """
        weight = reweight(input, self.cutoff)
        return fast_dca(input, weight, self.penalty)


class MSADataset(Dataset):
    """
    """
    init_keys = ["ac", "label"]
    # position_dict : Dict[int, Deque[int]] = {}
    # seqrecord_dict : Dict[int, Iterable[SeqRecord]] = {}
    file_type = "a3m"

    def __init__(self, file_address: str, working_dir:str,
                 num_classes: int, mode: str = "train", task: str = "bp",
                 top_k: int = 40, max_len: int = 2000,
                 msa_buffer_size: int = 10000,
                 targets : Optional[Dict] = None) -> None:
        """
        """
        self.file_address = file_address
        self.num_classes = num_classes
        self.mode = mode
        self.task = task
        self.working_dir = working_dir
        self.top_k = top_k
        self.MAXLEN = max_len
        self.msa_buffer_size = msa_buffer_size

        if targets is None:
            self.load_data()
        else:
            self.targets = targets

        self.len = len(self.targets["ac"])
        print(f"Loaded {mode}-set, X: {self.file_address}, length: {len(self)}")
    
    def __len__(self):
        return self.len
    
    def load_data(self, file_address : str = None, 
                  file_type=None) -> None:
        if file_address is not None:
            self.file_address = file_address
        
        if file_type is not None:
            self.file_type = file_type
        
        data = torch.load(self.file_address)
        data = data[self.mode] # training data or test data
        data = data[self.task] # cco, mfo or bpo

        self.targets = {k: data[k] for k in self.init_keys}
        
        msa_namefmt = "{}." + f"{self.file_type}"
        self.targets["msa"] = [os.path.join(self.working_dir, msa_namefmt.format(x))
                               for x in data["ac"]]

        self.init_keys = self.targets.keys()
        mask = [os.path.exists(x) for x in self.targets["msa"]]
        self.targets = {k: [x for i, x in enumerate(self.targets[k]) if mask[i]]
                        for k in self.init_keys}
    
    def get(self, key, index):
        return self.targets[key][index]
    
    # def get_seq_pos(self, index):
    #     p : int = self.position_dict[index].popleft()
    #     self.position_dict[index].append(p)
    #     return p
   
    def extract_from_a3m(self, msa_path : str):
        """
        ensure the size of sequence extracted from file is larger than top_k
        """
        seqrecord_gen, seq_record = parsing_a3m_by_range(msa_path, start=0,
                                                size_of_msa=self.msa_buffer_size,
                                                only_seq_record=True)

        def seqrecord2mat(seqrecord : Iterable[SeqRecord]):
            records = list(seqrecord)
            r_size = len(records)
            indices = np.random.permutation(r_size)
            sel_indices = indices[:self.top_k]
            mask = np.array([False] * self.msa_buffer_size, dtype=bool)
            mask[sel_indices] = True
            seqfilter = filter(select_from_generator_by_index(mask.tolist()), records)
            seq_mat = seqrecord2numpy(seqfilter)
            seq_mat = encoding(seq_mat)
            return seq_mat

        try:
            seq_mat = seqrecord2mat(seqrecord_gen)
            # print(msa_path, seq_mat.shape)
        except RuntimeError:
            # print("capture stopiteration exception")
            seq_mat = seqrecord2mat(seqrecord_gen)
        except StopIteration:
            # print("capture stopiteration exception")
            seq_mat = seqrecord2mat(seqrecord_gen)


        return seq_mat
    

    def __getitem__(self, index):
        # msa = np.load(self.get("msa",index))
        msa_path = self.get("msa",index)
        msa = self.extract_from_a3m(msa_path)
        # print(f"the type of msa is {type(msa)} and the shape is {msa.shape}")
        msa = torch.tensor(msa, dtype=torch.int64)
        # print(f"msa shape {msa.shape}")

        c, h = msa.shape

        pad_c = 0
        pad_h = 0
        if h < self.MAXLEN:
            pad_h = self.MAXLEN - h
        
        if c < self.top_k:
            pad_c = self.top_k - c
        
        if h < self.MAXLEN or c < self.top_k:
            msa = F.pad(msa, (0, pad_h, 0, pad_c))

        # if self.position_dict.get(index) is None:
        #     bound = h - self.MAXLEN
        #     pos_iter = range(0, bound, self.MAXLEN) if bound >= self.MAXLEN else [0]
        #     self.position_dict[index] = deque(pos_iter)
        
        # pos_h = self.get_seq_pos(index)
        # msa: k x MAXLEN
        # X = msa[:self.top_k, pos_h:pos_h+self.MAXLEN]
        X = msa[:self.top_k, :self.MAXLEN]

        # X = F.one_hot(X, num_classes=21)

        y = self.get("label",index)
        assert isinstance(y, Iterable), "y is not Iterable"

        if isinstance(y, Set):
            y = list(y)
        else:
            y = [t for t in y]
        
        assert isinstance(y, List), "y is not List"

        classes = np.arange(self.num_classes)
        y : np.ndarray = np.where(np.isin(classes, y), 1, 0)

        return X, y


if __name__ == "__main__":
    import time
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    mode = "train"
    file_path = "/home/Databases/GOA-S2F/CAFA/CAFA3/training/utils/GO_new_data.pt"
    if mode == "train":
        working_dir = "/home/Databases/GOA-S2F/CAFA/CAFA3/training/a3m_seqs" 
        batch_size = 24
    else:
        working_dir = "/home/Databases/GOA-S2F/CAFA/CAFA3/target/a3m_uniref"
        batch_size = 24
    
    strategy = "emb_plus_one_hot"
    cutoff = 0.8
    penalty = 4.5
    embedding_dim = 21
    
    msa_dataset = MSADataset(file_path,working_dir,21585,mode, "bp",
                             top_k=40,max_len=2000,msa_buffer_size=10000)

    msa_dataloader = torch.utils.data.DataLoader(msa_dataset, batch_size=batch_size,
                                                num_workers=10)

    pre_model = MSAEncoder(cutoff, penalty, embedding_dim,
                          encoding_strategy=strategy)
    # pre_model = nn.Embedding(21, embedding_dim)
    pre_model = torch.jit.script(pre_model)
    st = time.time()
    assert mode in ["train", "test"]
    Epochs = 2
    for epoch in range(Epochs):
        for i, (X, y) in enumerate(msa_dataloader):
            X = pre_model(X)
            ed = time.time()
            print(f"Epoch [{epoch+1}/{Epochs}]: consumed {ed-st}s for item {i}")
            print(f"X shape {X.shape}, y shape {y.shape}")
            print(np.where(y)[0].shape)
            st = time.time()


