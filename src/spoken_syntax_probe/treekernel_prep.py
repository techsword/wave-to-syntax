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


# Default workers: trust the environment — SLURM's allocation when running
# under SLURM (SLURM_CPUS_PER_TASK), otherwise the process's CPU affinity
# (sched_getaffinity). No hard-coded count; explicit n_jobs values are
# respected as given.
def default_n_jobs():
    env = os.environ.get('SLURM_CPUS_PER_TASK', '')
    if env.isdigit() and int(env) > 0:
        return int(env)
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 2)

# Tasks per joblib dispatch. ~10^4 test points become a few hundred
# dispatches, which cuts IPC overhead; Parallel preserves task order.
PARALLEL_BATCH_SIZE = 64


def effective_n_jobs(n_jobs):
    """Return a safe worker count; never joblib's all-cores sentinel (-1)."""
    return max(1, n_jobs) if n_jobs and n_jobs > 0 else default_n_jobs()


def compute_kernel(f, data1, trees_filtered, normalize = True, anchor_trees = None, anchor_self = None):
    """Kernel row for one test tree against the anchor trees.

    ``anchor_trees`` and ``anchor_self`` are optional precomputed delexed
    anchor trees and their self-kernels. When omitted, both are recomputed per
    call, which preserves the original standalone behaviour exactly.
    """
    tree1 = delex(data1[0][0])
    tree1_kern = f(tree1, tree1)
    kernel_container = []
    for j, data2 in enumerate(trees_filtered):
        if anchor_trees is None:
            tree2 = delex(data2[0][0])
            self2 = f(tree2, tree2)
        else:
            tree2 = anchor_trees[j]
            self2 = anchor_self[j]
        denom = (tree1_kern * self2)**0.5 if normalize else 1.0
        kernel_container.append(list((f(tree1, tree2)/denom, data1[-1],data2[-1])))
    return np.array(kernel_container)

def generate_kernel_regress(tree_paths, seed = 42, alpha = 0.5, num_anchors = 200, save_path = 'regress-data', normalization = True, parallel = False, rewrite = False, n_jobs = None):
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
            trees_filtered = [x for x in tree_list if len(str.split(x[1])) < word_upper_limit]
            trees_filtered = [[item, i] for i, item in enumerate(trees_filtered)][:]
            random.shuffle(trees_filtered)

            ref_pts = trees_filtered[:num_anchors]
            test_pts = trees_filtered[num_anchors+1:]
            # Anchor delex and self-kernels do not depend on the test tree, so
            # compute them once per dataset instead of once per test point.
            anchor_trees = [delex(t[0][0]) for t in ref_pts]
            anchor_self = [K(t, t) for t in anchor_trees]
            if parallel == True:
                from joblib import Parallel, delayed
                tree_kernel_container = Parallel(
                        n_jobs=effective_n_jobs(n_jobs), backend='loky',
                        batch_size=PARALLEL_BATCH_SIZE
                        )(delayed(compute_kernel)(
                            K, i, ref_pts, normalization, anchor_trees, anchor_self
                            ) for i in tqdm(test_pts))
            else:
                tree_kernel_container = []
                for test_pt in tqdm(test_pts):
                    tree_kernel_container.append(compute_kernel(
                        K, test_pt, ref_pts, normalization, anchor_trees, anchor_self))
                
            torch.save(tree_kernel_container, save_file)

    
if __name__ == "__main__":
    tree_paths = [x for x in os.listdir() if 'generated_trees' in x]
    generate_kernel_regress(tree_paths,parallel=True)