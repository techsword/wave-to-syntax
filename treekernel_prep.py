import json
import os
import random

import numpy as np
import torch
import ursa.util as U
from tqdm.auto import tqdm
from ursa.kernel import Kernel, delex

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import ursa.util as U
from nltk.tree import Tree


def compute_kernel(f, data1, trees_filtered, normalize = True):
    tree1 = delex(data1[0][0])
    tree1_kern = f(tree1, tree1)
    kernel_container = []
    for data2 in trees_filtered:
        tree2 = delex(data2[0][0])
        denom = (tree1_kern * f(tree2, tree2))**0.5 if normalize else 1.0
        kernel_container.append(list((f(tree1, tree2)/denom, data1[-1],data2[-1])))
    return np.array(kernel_container)

def generate_kernel_regress(tree_paths, seed = 42, alpha = 0.5, num_anchors = 200, save_path = 'regress-data', normalization = True, parallel = False, rewrite = False, subset = None):
    random.seed(seed)
    K = Kernel(alpha=alpha)
    for generated_tree in tree_paths:
        tree_kernel = []
        datasetname = '_'.join(generated_tree.split('_')[:2])
        save_file = os.path.join(save_path, datasetname + '_' + str(seed)+'_'+str(num_anchors)+'anchors' "_regress_kernel.pt")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if os.path.isfile(save_file) and not rewrite:
            print(f"{save_file} exists already, skipping!")
        else:
            tree_list = torch.load(generated_tree)
            word_upper_limit = 52 if 'libri' in generated_tree else 20
            tree_list = tree_list if subset == None else tree_list[:subset]
            trees_filtered = [x for x in tree_list if len(str.split(x[1])) < word_upper_limit]
            trees_filtered = [[item, i] for i, item in enumerate(trees_filtered)][:]
            random.shuffle(trees_filtered)

            ref_pts = trees_filtered[:num_anchors]
            test_pts = trees_filtered[num_anchors+1:]
            if parallel == True:
                from joblib import Parallel, delayed
                tree_kernel_container = Parallel(
                        n_jobs=-1, backend='loky'
                        )(delayed(compute_kernel)(K,i,ref_pts,normalization) for i in tqdm(test_pts))
            else:
                tree_kernel_container = []
                for test_pt in tqdm(test_pts):
                    tree_kernel_container.append(compute_kernel(K,test_pt, ref_pts, normalization))
                
            torch.save(tree_kernel_container, save_file)

    
if __name__ == "__main__":
    tree_paths = [x for x in os.listdir() if 'generated_trees' in x]
    generate_kernel_regress(tree_paths,parallel=True)