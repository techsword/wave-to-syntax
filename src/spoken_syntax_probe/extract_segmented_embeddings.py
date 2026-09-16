import gc
import math
import os
import random

import numpy as np
import pandas as pd
import textgrid
import torch
from tqdm import tqdm
import pandas as pd
import torchaudio
from torch.utils.data import DataLoader, Dataset


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# from .utils.custom_classes import Corpus
from .utils.custom_functions import loading_pretrained_model


def segment_audio_emb(emb, segment_df, audio_len, hidden_size=None):
    total_frames = emb.shape[1]
    segment_df['startFrame'] = (segment_df['startTime']/audio_len*total_frames).map(math.ceil)
    segment_df['endFrame'] = (segment_df['endTime']/audio_len*total_frames).map(math.ceil)
    segment_df = segment_df[~segment_df['transcription'].str.contains('sil', na = False)]
    segment_dict = segment_df.iloc[:,3:].to_dict()
    # Derive the hidden size from the layer tensor (or the model config passed
    # in). The old hard-coded 768 produced wrong-shaped rows for *-large models
    # (hidden size 1024) and crashed at assignment.
    if hidden_size is None:
        hidden_size = emb.shape[-1]
    segments = torch.zeros(len(segment_df), hidden_size)
    for i, x in enumerate(segment_dict['transcription']):
        # segment_tensor = 
        segment_start = segment_dict['startFrame'][x]
        segment_end = segment_dict['endFrame'][x]
        segment_tensor = torch.mean(emb[:,segment_start:segment_end,:].cpu().squeeze(),0,True)
        # print(segment_tensor.shape)
        segments[i] = segment_tensor.clone()
    return segments


def generating_features(dataset, model, aligned_path, layer = 12, sr = 16000):
    feat_list = []
    annot_list = []
    audio_len_list = []
    if 'libri' in aligned_path.lower():
        # dataset = [x for x in dataset if len(str.split(x[1])) < 52]
        len_ceil = 52
    elif 'spokencoco' in aligned_path.lower():
        # dataset = [x for x in dataset if len(str.split(x[1])) < 20]
        len_ceil = 20
    print(f'wordcount ceiling set to {len_ceil}! now start iterating through the dataset!')
    for waveform, annot, audio_file, csv_file in tqdm(dataset):
        if len(str.split(annot)) > len_ceil:
            continue
            
        total_frames = waveform.shape[1]
        segment_df = pd.read_csv(csv_file)
        audio_len = total_frames/sr
        annot_list.append(annot)
        audio_len_list.append(audio_len)

        with torch.inference_mode():
            features, _ = model.to(device).extract_features(waveform.to(device), num_layers = layer)
            # Restore word segmentation before stacking. The published
            # segmented artifacts store one mean embedding per aligned word,
            # not raw frame-level stacks.
            hidden_size = getattr(getattr(model, 'config', None), 'hidden_size', None)
            features = [segment_audio_emb(x, segment_df, audio_len, hidden_size) for x in features]
            features = torch.stack(features).detach().cpu().numpy()

            # return features[0], segment_df, audio_len
            # features = features[-1]
            # features = [torch.mean(x.cpu(),dim=1).squeeze().numpy() for x in features]
            # feat_list.append(torch.mean(features,dim=1).squeeze().numpy())
            feat_list.append(features)
    print(f"there are {len(annot_list)} in the extracted dataset")
    return list(zip(feat_list,annot_list, audio_len_list))






class WordSegmentedCorpus(Dataset):

    def __init__(self, csv_file, root_dir, aligned_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the Librispeech directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.audiofilelist = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.aligned_path = aligned_path
        self.transform = transform

    def __len__(self):
        return len(self.audiofilelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root_dir,
                                self.audiofilelist.iloc[idx, 0])
        audio, sr = torchaudio.load(audio_name)
        annot = self.audiofilelist.iloc[idx,-1]
        # sample = {'audio': audio.to(device), 'file': audio_name, 'sr': sr, 'annot': sent}
        audio = audio.to(device)
        if 'libri' in audio_name.lower():
            alignment_file =  os.path.join(self.aligned_path, self.audiofilelist.iloc[idx,0].split('.')[0]+'.csv')
        elif 'spokencoco' in audio_name.lower():
            alignment_file = os.path.join(self.aligned_path,"/".join(self.audiofilelist.iloc[idx,0].split('.')[0].split('/')[-2:])+'.csv')


        if self.transform:
            audio = self.transform(audio)

        return audio, annot, audio_name, alignment_file


class ObservationIterator(Dataset):
    """ List Container for lists of Observations and labels for them.
    Used as the iterator for a PyTorch dataloader.
    """

    def __init__(self, observations, task):
        self.observations = observations
        self.set_labels(observations, task)

    def set_labels(self, observations, task):
        """ Constructs aand stores label for each observation.
        Args:
        observations: A list of observations describing a dataset
        task: a Task object which takes Observations and constructs labels.
        """
        self.labels = []
        for observation in tqdm(observations, desc='[computing labels]'):
            self.labels.append(task.labels(observation))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.labels[idx]

class EmbsDataset():
    def __init__(self):
        pass
    def custom_pad(self, batch):
        '''Pads sequences with 0 and labels with -1; used as collate_fn of DataLoader.
        
        Loss functions will ignore -1 labels.
        If labels are 1D, pads to the maximum sequence length.
        If labels are 2D, pads all to (maxlen,maxlen).
        Args:
        batch_observations: A list of observations composing a batch
        
        Return:
        A tuple of:
            input batch, padded
            label batch, padded
            lengths-of-inputs batch, padded
            Observation batch (not padded)
        '''
        seqs, annot, audio_name, alignment_file = zip(*batch)

        lengths = torch.tensor([len(x) for x in seqs]).to(device)
        seqs = pad_sequence(seqs, batch_first=True)
        label_shape = batch[1].shape
        maxlen = int(torch.max(lengths))
        label_maxshape = [maxlen for x in label_shape]
        labels = [-torch.ones(*label_maxshape).to(device) for x in seqs]
        for index, x in enumerate(batch):
            length = x[1].shape[0]
            if len(label_shape) == 1:
                labels[index][:length] = x[1]
            elif len(label_shape) == 2:
                labels[index][:length,:length] = x[1]
            else:
                raise ValueError("Labels must be either 1D or 2D right now; got either 0D or >3D")
        labels = torch.stack(labels)
        return seqs, labels, lengths, batch


from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    seq, annot, audio_name, alignment_file = zip(*batch)

    seq = [x.squeeze() for x in seq]
    seq_batched = pad_sequence(seq, batch_first=True)
    lengths = torch.tensor([len(x)/16000 for x in seq])
    return seq_batched, lengths, annot, audio_name, alignment_file


def model_tag_from_path(model_path):
    '''Short model tag used in output file names. Strips a trailing .pt from
    local checkpoint paths; for Hugging Face ids uses the repo name.'''
    tag = os.path.basename(str(model_path).rstrip('/'))
    return tag[:-3] if tag.endswith('.pt') else tag


def main(model, dataset, model_path, save_dir='segmented_embeddings', csv_file=None, root_dir=None, aligned_path=None, model_tag=None):
    # Dataset roots and aligned dirs are machine-local and must be provided
    # (see README "Datasets"). csv_file defaults to the file generated into the
    # repo root by preprocessing.py; save_dir is relative to the repo.
    tag = model_tag or model_tag_from_path(model_path)
    if dataset == 'scc':
        csv_file = csv_file or 'spokencoco_val.csv'
        save_file = os.path.join(save_dir, tag + '_spokencoco.pt')
    elif dataset == 'libri':
        csv_file = csv_file or 'librispeech_train-clean-100.csv'
        save_file = os.path.join(save_dir, tag + '_librispeech.pt')
    else:
        raise KeyError(f'dataset {dataset!r} is not supported')

    if root_dir is None or aligned_path is None:
        raise ValueError(
            'root_dir and aligned_path must be provided: they point to the audio '
            'files and their word-aligned TextGrid-derived csvs. Both are '
            'machine-local and no longer hardcoded; pass --root and --aligned_path '
            '(see README "Datasets").')

    data = WordSegmentedCorpus(csv_file, root_dir, aligned_path)
    if os.path.isfile(save_file):
        print(f"{save_file} exists already! not overwriting and skipped")
    else:
        os.makedirs(save_dir, exist_ok=True)
        print(f"extracting segmented embeddings and saving to {save_file}")
        extracted_features = generating_features(data, model, aligned_path)

        # Protocol 5 legacy serialization keeps the format load-identical but
        # avoids the ~5.5x peak-memory overhead (and >4 GiB OverflowError) of
        # pickle protocol 2 on many-small-array artefacts. Validated on the
        # workstation replication.
        torch.save(extracted_features, save_file,
                   pickle_protocol=5, _use_new_zipfile_serialization=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Extract word-segmented embeddings from a wav2vec 2.0 model '
                    'loaded with transformers.')
    parser.add_argument('--model_path', default=None,
                        help='Hugging Face model id (e.g. facebook/wav2vec2-base) '
                             'or a local HF checkpoint directory. Required.')
    parser.add_argument('--model_tag', default=None,
                        help='Output-file tag; defaults to the basename of '
                             '--model_path with a trailing .pt stripped. Pass '
                             'e.g. wav2vec_small (or checkpoint-10000) to '
                             'reproduce the published segmented-embedding file '
                             'names.')
    parser.add_argument('--dataset', choices=['scc', 'libri'], default='scc',
                        help='Which corpus to extract from (scc=SpokenCOCO, libri=LibriSpeech).')
    parser.add_argument('--csv', default=None,
                        help='Dataset csv (defaults to spokencoco_val.csv or '
                             'librispeech_train-clean-100.csv, generated by preprocessing.py).')
    parser.add_argument('--root', default=None,
                        help='Root directory of the audio files. Required.')
    parser.add_argument('--aligned_path', default=None,
                        help='Directory of word-aligned csvs (from forced_alignment.py). Required.')
    parser.add_argument('--save_dir', default='segmented_embeddings',
                        help='Directory to save the extracted .pt files.')
    cli = parser.parse_args()

    if cli.model_path is None:
        parser.error('--model_path is required (HF model id or local HF checkpoint).')

    model = loading_pretrained_model(cli.model_path).to(device)
    main(model, cli.dataset, cli.model_path, cli.save_dir, cli.csv, cli.root, cli.aligned_path, cli.model_tag)
