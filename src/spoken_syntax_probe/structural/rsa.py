import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import gc
import os

import ursa.util as U
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm.auto import tqdm

from ..treekernelprobe import kernel_dataset_name


def pairwise_distance_calc(embedding_file,pd_save_path, rewrite = False, device = 'cuda'):
    '''
    embedding_file: the file containing embeddings and annotations
    '''

    embeddings, _, annot, _,_,_ = torch.load(embedding_file)
    datasetname = embedding_file.split('_',1)[1].replace('_extracted.pt','')
    ceil = 52 if 'libri' in datasetname else 20

    if isinstance(annot[0], tuple):
        filter_mask = np.array([i for i, x in enumerate(annot) if len(str.split(x[0])) < ceil])
    else:
        filter_mask = np.array([i for i, x in enumerate(annot) if len(str.split(x)) < ceil])
    filtered_embs = np.take(np.array(embeddings),filter_mask, axis=0).astype(float)

    modelname = os.path.basename(embedding_file).split('_',1)[0]
    datasetname = embedding_file.split('_',1)[1].replace('_extracted.pt','')
    tqdm.write(f"calculating pairwise distances for {modelname, datasetname}")

    for layer in tqdm(range(filtered_embs.shape[1])):
        save_file = '_'.join([modelname, datasetname, str(layer)])+"_pd.pt"
        if os.path.isfile(os.path.join(pd_save_path,save_file)) and not rewrite:
            tqdm.write(f"{save_file} exists already! skipping to the next one!")
        else:
            tqdm.write(f'pulling embedding from layer {layer} and saving to {save_file}')
            layer_embeddings = torch.tensor(filtered_embs[:,layer,:])
            layer_similarity = pairwise_cosine_similarity(layer_embeddings.to(device)).detach().cpu().numpy()
            # pairwise_distance_container.append(layer_similarity)
            # Protocol 5 legacy serialization keeps the format load-identical
            # but avoids the ~5.5x peak-memory overhead (and >4 GiB
            # OverflowError) of pickle protocol 2 on many-small-array
            # artefacts. Validated on the workstation replication.
            torch.save(layer_similarity, os.path.join(pd_save_path, save_file),
                       pickle_protocol=5, _use_new_zipfile_serialization=False)
        # pairwise_distance_container = np.stack(pairwise_distance_container)
        # return pairwise_distance_container

def compute_pairwise_dist_for_embs(embedding_path = 'embeddings', pd_save_path = 'pairwise_distances', rewrite = False, device = 'cuda'):
    emb_files = [os.path.join(embedding_path, x) for x in os.listdir(embedding_path) if 'extracted.pt' in x]    
    for embedding_file in tqdm(emb_files):
        if 'BOW' in embedding_file:
            device = 'cpu'
        else:
            device = device
        pairwise_distance_calc(embedding_file, pd_save_path, rewrite, device)
        # pairwise_distances = pairwise_distance_calc(embedding_file, rewrite)
        # torch.save(pairwise_distances,os.path.join(pd_save_path,save_file), pickle_protocol=4)
        # print(f"{os.path.join(pd_save_path,save_file)} is saved!")



def pearson_r_score(Y_true, Y_pred): 
     r =  U.pearson_r(Y_true, Y_pred, axis=0).mean() 
     return r


def pairwise_files_for_kernel(kern, pairwise_distance_files):
    """Pairwise-distance files that match a tree kernel's corpus.

    Kernel files are named ``spokencoco_val_*`` or ``librispeech_train_*``.
    ``kernel_dataset_name`` maps both the current names and the legacy ``scc``
    token, so the result is always defined. The old ``'scc' in kern`` test left
    ``list_of_files`` unbound for ``spokencoco_val`` kernels.
    """
    dataset_token = 'spokencoco' if kernel_dataset_name(kern) == 'spokencoco' else 'libri'
    return [x for x in pairwise_distance_files if dataset_token in x]


def parse_pd_filename(pd_file):
    """Parse ``(modelname, datasetname, layer)`` from a ``*_<layer>_pd.pt`` name."""
    parts = os.path.basename(pd_file).split('_')
    return '_'.join(parts[:-4]), '_'.join(parts[-4:-2]), int(parts[-2])


def kernel_layer_pearson(kernel, distance_matrix, kernel_pairs):
    """Pearson r between kernel similarities and the matching pairwise distances.

    ``distance_matrix`` is one layer's ``(N, N)`` pairwise-similarity matrix and
    ``kernel_pairs`` an ``(P, 2)`` array of ``(test, ref)`` index pairs aligned
    with ``kernel``.
    """
    layer_distance = np.array([distance_matrix[i, j] for i, j in kernel_pairs])
    return pearson_r_score(kernel, layer_distance)

def kernel_matches_run(kernel_path, seed, delexed, alpha):
    """Whether a kernel file belongs to this (seed, delexed, alpha) run.

    ``treekernel_prep`` regress kernels are named
    ``<dataset>_<seed>_<n>anchors_regress_kernel.pt`` and carry no
    ``_delexed``/alpha suffix, so they match on seed alone. Legacy names keep
    the original seed / ``_delexed`` / alpha match.
    """
    name = os.path.basename(kernel_path)
    if 'anchors_regress_kernel' in name:
        return str(seed) in name
    if ('_delexed' in name) != bool(delexed):
        return False
    return str(seed) in name and str(alpha) in name


def load_kernel_pairs(kernel_path):
    """Load ``(kernel values, (test, ref) index pairs)`` from a kernel file.

    ``treekernel_prep`` regress kernels store a list of per-test-point
    ``(n_anchors, 3)`` arrays; the legacy format stores one stacked ``(P, 3)``
    array.
    """
    loaded = torch.load(kernel_path)
    if 'anchors_regress_kernel' in os.path.basename(kernel_path):
        rows = np.vstack(loaded)
    else:
        rows = np.array(loaded, dtype=object)
    return np.array(rows[:, 0], dtype=float), np.array(rows[:, 1:]).astype(int)


def main(alpha = 0.5, seed = 42, 
         delexed = True, 
         tree_kernel_path = 'tree_kernel',
         pairwise_distance_path = 'pairwise_distances'
             ):
    alpha = str(alpha)
    seed = str(seed)
    kernel_files = [os.path.join(tree_kernel_path,x) for x in os.listdir(tree_kernel_path) if '_kernel' in x]
    pairwise_distance_files = [os.path.join(pairwise_distance_path,x) for x in os.listdir(pairwise_distance_path)]
    tree_kernel_files = [x for x in kernel_files if kernel_matches_run(x, seed, delexed, alpha)]
    for kern in tqdm(tree_kernel_files):
        tqdm.write(f'using tree kernel {os.path.basename(kern)}')
        kernel, kernel_pairs = load_kernel_pairs(kern)
        list_of_files = pairwise_files_for_kernel(kern, pairwise_distance_files)

        for x in tqdm(list_of_files):
            tqdm.write(f'using pairwise distance file {os.path.basename(x)}')
            modelname, datasetname, layer = parse_pd_filename(x)
            # Each *_pd.pt holds one layer's (N, N) pairwise-similarity matrix,
            # so the layer index comes from the file name, not the array shape.
            calculated_distances = torch.load(x)
            r_score = kernel_layer_pearson(kernel, calculated_distances, kernel_pairs)
            tqdm.write(f'{modelname} {datasetname} layer {layer} alpha {alpha}: pearson_r={r_score}')
            del calculated_distances
            gc.collect()




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Representational Similarity Analysis (RSA) for the syntax probe.')
    parser.add_argument('--embedding_path', default='embeddings',
                        help='Directory of extracted embeddings (from embedding_generation.py).')
    parser.add_argument('--pd_save_path', default='pairwise_distances',
                        help='Directory to save pairwise-distance tensors.')
    parser.add_argument('--tree_kernel_path', default='tree_kernel',
                        help='Directory of tree-kernel tensors (from treekernel_prep.py).')
    parser.add_argument('--pairwise_distance_path', default='pairwise_distances',
                        help='Directory of pairwise-distance tensors to score against tree kernels.')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--delexed', action=argparse.BooleanOptionalAction, default=True,
                        help='Use delexicalized kernels (default: on).')
    parser.add_argument('--run_rsa', action='store_true',
                        help='Also run the RSA correlation scoring (default only computes '
                             'pairwise distances, matching the original behaviour).')
    cli = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    compute_pairwise_dist_for_embs(embedding_path=cli.embedding_path,
                                   pd_save_path=cli.pd_save_path,
                                   device=device)
    if cli.run_rsa:
        main(alpha=cli.alpha, seed=cli.seed, delexed=cli.delexed,
             tree_kernel_path=cli.tree_kernel_path,
             pairwise_distance_path=cli.pairwise_distance_path)
