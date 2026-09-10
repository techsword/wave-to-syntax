import os

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary


class Corpus(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audiofilelist = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.audiofilelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_name = self.audiofilelist.iloc[idx]['path']
        audio, sr = torchaudio.load(audio_name)
        annot = self.audiofilelist.iloc[idx]['annot']
        depth = self.audiofilelist.iloc[idx]['depth']
        audiolen = audio.shape[-1]/sr
        wordcount = len(annot.split())


        return audio, annot, depth, audiolen, wordcount, audio_name

    def collate(self, batch):
        return batch
    
class textCorpus(Dataset):

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audiofilelist = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.audiofilelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_name = self.audiofilelist.iloc[idx]['path']
        annot = self.audiofilelist.iloc[idx]['annot']
        depth = self.audiofilelist.iloc[idx]['depth']


        return annot, depth, audio_name

    def collate(self, batch):
        return batch
    
class textCorpus_no_depth(Dataset):

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audiofilelist = pd.read_csv(csv_file,header=None)

    def __len__(self):
        return len(self.audiofilelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        audio_name = self.audiofilelist.iloc[idx][0]
        annot = self.audiofilelist.iloc[idx][1]
        return annot, audio_name