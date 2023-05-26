import json
import os
import random

import numpy as np
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm.auto import tqdm
from ursa.regress import Regress

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import ursa.util as U
from nltk.tree import Tree


def load_tree_kernel(tree_kernel_file):
    tree_kernel_set = np.vstack(torch.load(tree_kernel_file))
    tree_kernel_set = tree_kernel_set[np.lexsort((tree_kernel_set[:,2], tree_kernel_set[:,1]))]
    tk = tree_kernel_set[:,0]
    test_sent_idx = np.unique(tree_kernel_set[:,1])
    ref_sent_idx = np.unique(tree_kernel_set[:,2])
    tk = tk.reshape(test_sent_idx.shape[0], ref_sent_idx.shape[0])
    word_upper_limit = 52 if 'libri' in tree_kernel_file else 20

    return tk, ref_sent_idx, test_sent_idx, word_upper_limit


def load_embs(embedding_file, ceil,ref_sent_idx, test_sent_idx):
    embeddings, _, annot, _,_,_ = torch.load(embedding_file)
    modelname = os.path.basename(embedding_file).split('_',1)[0]
    datasetname = embedding_file.split('_',1)[1].replace('_extracted.pt','')
    if isinstance(annot[0], tuple):
        filter_mask = np.array([i for i, x in enumerate(annot) if len(str.split(x[0])) < ceil])
    else:
        filter_mask = np.array([i for i, x in enumerate(annot) if len(str.split(x)) < ceil])
    filtered_embs = np.take(np.array(embeddings),filter_mask, axis=0)
    ref_embs = np.take(filtered_embs, ref_sent_idx.astype(int), axis = 0)
    test_embs = np.take(filtered_embs, test_sent_idx.astype(int), axis = 0)

    return modelname, datasetname, ref_embs, test_embs
    
def run_probe():
    tk_data_path = 'regress-data/'
    tree_kernel_paths = [os.path.join(tk_data_path,x) for x in os.listdir(tk_data_path) if 'anchors_regress_kernel' in x]

    embedding_path = 'embeddings'
    all_embedding_files = [x for x in os.listdir(embedding_path) if '.pt' in x]
    embedding_files = [os.path.join(embedding_path,x) for x in all_embedding_files]
    tqdm.write(f'looking at {embedding_files}')
    for tree_kernel_file in tqdm(tree_kernel_paths):
        tk, ref_sent_idx, test_sent_idx, ceil = load_tree_kernel(tree_kernel_file)
        datasetname = '_'.join(os.path.basename(tree_kernel_file).split("_")[:2])
        for embedding_file in tqdm([x for x in embedding_files if datasetname in x]):
            modelname, datasetname, ref_embs, test_embs = load_embs(embedding_file, ceil, ref_sent_idx, test_sent_idx)
            if modelname == 'BOW':
                device = 'cpu'
            else:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            for layer in tqdm(range(ref_embs.shape[1])):
                    R = Regress()
                    test = torch.tensor(test_embs[:,layer,:], dtype = float).to(device)
                    ref = torch.tensor(ref_embs[:,layer,:], dtype = float).to(device)
                    pd = pairwise_cosine_similarity(test,ref).detach().cpu().numpy()
                    pd = np.nan_to_num(pd, copy=True, nan=1.0, posinf=None, neginf=None)
                    score = R.fit_report(X=pd, Y = tk)

                    result = {
                                'modelname': modelname,
                                'datasetname': datasetname,
                                'layer': layer
                            }
                    print(result|score)




if __name__ == "__main__":

    run_probe()