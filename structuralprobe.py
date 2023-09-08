import argparse
import math
import os
import random
import sys
from itertools import islice

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import textgrid
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm.auto import tqdm
from utils.struct_probe_utils import (L1DepthLoss, L1DistanceLoss,
                                     OneWordPSDProbe, TwoWordPSDProbe,
                                     WordPairReporter, WordReporter)

### There are nan results in some spearmanr correlations. this was resolved by changing the mean function in reporter to nanmean.
### This is a trimmed version of structural probe from https://github.com/john-hewitt/structural-probes 
### Need significant improvement

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('sentencizer', before='parser')



class LoadFromDisk_(Dataset):

    def __init__(self, data_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with the Librispeech directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.emb, self.annot, self.audio_len = list(zip(*data_path))
        self.transform = transform

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):

        text = self.annot[idx]

        audio_len = self.audio_len[idx]
        text_len = len(text.split(' '))
        embedding = torch.stack([torch.tensor(x).clone().detach() for x in self.emb[idx]])


        if self.transform:
            pass
        return embedding, text, audio_len, text_len

    def collate(self, batch):
        return batch



def get_dep_distance_matrix(sent):
    doc = nlp(sent)
    edges = []
    for token in doc:
        for child in token.children:
            edges.append(('{0}'.format(token.lower_),
                          '{0}'.format(child.lower_)))
    graph = nx.Graph(edges)
    M = torch.zeros((len(doc), len(doc)))
    N = torch.zeros((len(doc)))
    for i, d1 in enumerate(doc):
        N[i] = nx.shortest_path_length(graph, source=d1.text.lower(), target = d1.sent.root.text.lower())
        for j, d2 in enumerate(doc):
            if  i > j: # No need to re-compute lower triangular
                M[i, j] = M[j, i]
            else:
                M[i, j] = nx.shortest_path_length(graph, source = str(d1).lower(), target = str(d2).lower())
    return M, N

def gen_labels(seg_embs, save_file = 'structural_probe_spokencoco_labels.pt'):
    if os.path.isfile(save_file):
        container = torch.load(save_file)
    else:
        container = {}
        for i,(_,sent,_,_) in enumerate(tqdm(seg_embs)):
            
            twd,wd = get_dep_distance_matrix(sent)
            container[i] = {'sent':sent, 'twd': twd, 'wd':wd}
        torch.save(container, save_file)
    return container

def load_labels(seg_embs, labels, mode = 'twd'):
    label = [labels[i][mode] for i in labels]
    return [entry+(label[i],) for i,entry in enumerate(seg_embs)]

def custom_pad(batch):
    seqs, _, _, _, label, = zip(*batch)
    # seqs = [x[0].clone().detach().to(device) for x in batch]
    lengths = torch.tensor([x.shape[0] for x in seqs], device=(device))
    seqs = nn.utils.rnn.pad_sequence(seqs, batch_first=True).to(device)
    label_shape = label[0].shape
    maxlen = int(torch.max(lengths))
    label_maxshape = [maxlen for x in label_shape]
    labels = [-torch.ones(*label_maxshape, device=device) for x in seqs]

    for index, x in enumerate(batch):
        length = x[-1].shape[0]
        if len(label_shape) == 1:
            labels[index][:length] = x[-1]
        elif len(label_shape) == 2:
            labels[index][:length,:length] = x[-1]
    labels = torch.stack(labels)
    return seqs, labels, lengths, batch

def train_until_convergence(probe, loss, train_dataset, dev_dataset, layer = 0, max_epochs = 30, params_path = 'predictor.params', save_dir = 'sp_data/'):
    """ Trains a probe until a convergence criterion is met.
    Trains until loss on the development set does not improve by more than epsilon
    for 5 straight epochs.
    Writes parameters of the probe to disk, at the location specified by config.
    Args:
      probe: An instance of probe.Probe, transforming model outputs to predictions
      model: An instance of model.Model, transforming inputs to word reprs
      loss: An instance of loss.Loss, computing loss between predictions and labels
      train_dataset: a torch.DataLoader object for iterating through training data
      dev_dataset: a torch.DataLoader object for iterating through dev data
    """
    torch.autograd.set_detect_anomaly(True)
    optimizer = optim.Adam(probe.parameters(), lr=0.001,eps = 1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,patience=0)

    min_dev_loss = sys.maxsize
    min_dev_loss_epoch = -1
    for epoch_index in tqdm(range(max_epochs), desc='[training]'):
      epoch_train_loss = 0
      epoch_dev_loss = 0
      epoch_train_epoch_count = 0
      epoch_dev_epoch_count = 0
      epoch_train_loss_count = 0
      epoch_dev_loss_count = 0
      for batch in tqdm(train_dataset, desc='[training batch]'):
        probe.train()
        optimizer.zero_grad()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = observation_batch
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        if not torch.isnan(batch_loss):
          batch_loss.backward()
          nn.utils.clip_grad_norm_(probe.parameters(), max_norm=2.0, norm_type=2)
        
          epoch_train_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
          epoch_train_epoch_count += 1
          epoch_train_loss_count += count.detach().cpu().numpy()
          optimizer.step()
        # print(batch_loss, count)
      for batch in tqdm(dev_dataset, desc='[dev batch]'):
        optimizer.zero_grad()
        probe.eval()
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = observation_batch
        predictions = probe(word_representations)
        batch_loss, count = loss(predictions, label_batch, length_batch)
        # print(batch_loss, count)
        # break
        if not torch.isnan(batch_loss):

          epoch_dev_loss += batch_loss.detach().cpu().numpy()*count.detach().cpu().numpy()
          epoch_dev_loss_count += count.detach().cpu().numpy()
          epoch_dev_epoch_count += 1
          
      scheduler.step(epoch_dev_loss)
      tqdm.write('[epoch {}] Train loss: {}, Dev loss: {}'.format(epoch_index, epoch_train_loss/epoch_train_loss_count, epoch_dev_loss/epoch_dev_loss_count))

      if epoch_dev_loss / epoch_dev_loss_count < min_dev_loss - 0.0001:
        params_path_final = os.path.join(save_dir, params_path)
        torch.save(probe.state_dict(), params_path_final)
        min_dev_loss = epoch_dev_loss / epoch_dev_loss_count
        min_dev_loss_epoch = epoch_index
        tqdm.write('Saving probe parameters')
      elif min_dev_loss_epoch < epoch_index - 4:
        tqdm.write('Early stopping')
        break
      

def predict(probe, dataset):
    """ Runs probe to compute predictions on a dataset.

    Args:
        probe: An instance of probe.Probe, transforming model outputs to predictions
        model: An instance of model.Model, transforming inputs to word reprs
        dataset: A pytorch.DataLoader object 

    Returns:
        A list of predictions for each batch in the batches yielded by the dataset
    """
    probe.eval()
    predictions_by_batch = []
    for batch in tqdm(dataset, desc='[predicting]'):
        observation_batch, label_batch, length_batch, _ = batch
        word_representations = observation_batch
        predictions = probe(word_representations)
        predictions_by_batch.append(predictions.detach().cpu().numpy())
    return predictions_by_batch

def run_report_results(probe, dataset, reporter, probe_params_path = 'layer0_predictor.params', save_dir = 'sp_data/'):
    """
    Reports results from a structural probe according to args.
    By default, does so only for dev set.
    Requires a simple code change to run on the test set.
    """
    probe.load_state_dict(torch.load(os.path.join(save_dir,probe_params_path)))
    probe.eval()

    dev_predictions = predict(probe = probe, dataset = dataset)
    split = '_'.join(probe_params_path.split('.')[:-2])
    reporter(dev_predictions, dataset, split)

def find_nans():
    from collections import defaultdict

    from scipy.stats import pearsonr, spearmanr
    args = {'device': 'cuda','hidden_dim':768, 'probe_training':{'epochs':30}, 'reporting':{'root': 'now', 'reporting_methods': ['spearmanr']}}
    probe = TwoWordPSDProbe(args)
    probe.load_state_dict(torch.load('sp_data/wav2vec_small.spokencoco.wd.layer_6.predictor.params'))
    probe.eval()
    scc_seg = LoadFromDisk_(torch.load('segmented_embeddings/wav2vec_small_spokencoco.pt'))
    labels = gen_labels(scc_seg)
    scc_observations_raw = load_labels(scc_seg, labels, mode = 'twd')  
    scc_observations = [x for x in scc_observations_raw if x[0].shape[1]==x[3]] 
    layer = 6
    layer_scc_observations = [tuple([x[0][layer,:,:]]+list(x[1:])) for x in scc_observations]
    train_data, test_data = train_test_split(layer_scc_observations,test_size=0.2, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=custom_pad, shuffle=False)
    dev_predictions = predict(probe, test_dataloader)
    lengths_to_spearmanrs = defaultdict(list)

    for prediction_batch, (data_batch, label_batch, length_batch,batch) in zip(dev_predictions, test_dataloader):
        for prediction, label, length in zip(prediction_batch, label_batch,length_batch):
            length = int(length)
            prediction = prediction[:length,:length]
            label = label[:length,:length].cpu()
            spearmanrs = [spearmanr(pred, gold) for pred, gold in zip(prediction, label)]
            if np.isnan(spearmanrs).any():
                print([x[1] for x in batch])
            lengths_to_spearmanrs[length].extend([x.correlation for x in spearmanrs])
    {length: np.mean(lengths_to_spearmanrs[length]) for length in lengths_to_spearmanrs}
    
def main(args, emb_file, mode = 'twd'):
    
    scc_ds = torch.load(emb_file)
    datasetname = os.path.basename(emb_file).split('_')[-1][:-3]
    modelname = '_'.join(os.path.basename(emb_file).split('_')[:-1])

    scc_seg = LoadFromDisk_(scc_ds)
    labels = gen_labels(scc_seg)
    scc_observations_raw = load_labels(scc_seg, labels, mode = mode)  
    scc_observations = [x for x in scc_observations_raw if x[0].shape[1]==x[3]] ### Filter out misaligned data entries
    if mode =='twd':
        probe_ = TwoWordPSDProbe(args)
        loss_fn = L1DistanceLoss(args)
        reporter_fn = WordPairReporter(args)
    elif mode == 'wd':
        probe_ = OneWordPSDProbe(args)
        loss_fn = L1DepthLoss(args)
        reporter_fn = WordReporter(args)
    else:
        raise KeyError(f'{mode} is not correct')

    for layer in tqdm(range(scc_observations[0][0].shape[0]),desc='[layers]'):
        probe_params_path = '.'.join([modelname, datasetname, mode, 'layer_'+str(layer), "predictor.params"])

        
        layer_scc_observations = [tuple([x[0][layer,:,:]]+list(x[1:])) for x in scc_observations]
        train_data, test_data = train_test_split(layer_scc_observations,test_size=0.2, shuffle=False)
        train_dataloader = DataLoader(train_data, batch_size=32, collate_fn=custom_pad, shuffle=False)
        test_dataloader = DataLoader(test_data, batch_size=32, collate_fn=custom_pad, shuffle=False)


        if not os.path.isdir(args['probe_training']['save_dir']):
            os.mkdir(args['probe_training']['save_dir'])
        save_dir_model = os.path.join(args['probe_training']['save_dir'],modelname)
        if not os.path.isdir(save_dir_model):
            os.mkdir(save_dir_model)

        train_until_convergence(probe   = probe_, 
                                loss    = loss_fn, 
                                train_dataset=train_dataloader, 
                                dev_dataset=test_dataloader, 
                                max_epochs=30, 
                                layer=layer, 
                                params_path = probe_params_path,
                                save_dir = save_dir_model)
                                # save_dir=args['probe_training']['save_dir'])
        if not os.path.isdir(args['reporting']['root']):
            os.mkdir(args['reporting']['root'])
        run_report_results(probe = probe_, 
                           dataset = test_dataloader, 
                           reporter= reporter_fn,
                           probe_params_path=probe_params_path,
                           save_dir=save_dir_model)
                        #    save_dir=args['probe_training']['save_dir'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Choose which extracted segmented embedding file to run structural probe on')
    parser.add_argument('dataset_num', nargs='?', default=0)
    parser.add_argument('mode', nargs='?', default= 'twd')
    parser.add_argument('--embedding_file', nargs='?', 
                        default='segmented_embeddings/wav2vec_small_spokencoco.pt', 
                        help='choose the embedding file')

    args = {'device': 'cuda','probe_training':{'epochs':30, 'save_dir': 'sp_data/'}, 'reporting':{'root': 'structural-probe-results', 'reporting_methods': ['spearmanr']}}


    cli_args = parser.parse_args()
    
    emb_file = cli_args.embedding_file
    args['hidden_dim'] = 1024 if 'large' in emb_file else 768
    mode = cli_args.mode
    main(args, emb_file, mode)

