import os
import re

import numpy as np
import pandas as pd
import stanza
import torch
from nltk import Tree
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from utils.custom_classes import textCorpus, textCorpus_no_depth
from utils.custom_functions import read_json_save_csv, walk_librispeech_dirs

device = 'cuda' if torch.cuda.is_available() else 'cpu'
sr = 16000

def make_dataset_csv(csv_file, dataset_root_dir, batch_size = 400, rewrite = False):
    """Use stanza to generate constituency trees, 
    save both trees and depth info for downstream experiments

    Args:
        csv_file (str): csv_file generated in prepreprocess()
        dataset_root_dir (str): same dataset root dirs as in prepreprocess
        batch_size (int, optional): Use larger batch size to process more sentences at the same time. Defaults to 400.
        rewrite (bool, optional): If rewrite or not. Defaults to False.
    """
    dataset_root_dir = os.path.expanduser(dataset_root_dir)
    save_file = "dataset_"+csv_file
    dataset_ID = csv_file.split(".")[0].split("-")[0]
    dataset_split = csv_file.split(".")[0].split('_')[-1]
    tree_save_file = dataset_ID+"_generated_trees.pt"
    if os.path.isfile(tree_save_file) and os.path.isfile(save_file) and not rewrite:
        print(f"{save_file} exists already! skipping generation")
    else:
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
        dataset = textCorpus_no_depth(csv_file)
        data, list_of_trees = [], []
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        for batch in tqdm(dataloader):
            annots, paths = batch
            in_docs = [stanza.Document([], text=d) for d in annots]
            out_docs = nlp(in_docs)
            for i, doc in enumerate(out_docs):
                datum_entry = {}
                datum_entry['path'] = os.path.join(dataset_root_dir,paths[i])
                datum_entry['annot'] = doc.text
                datum_entry['depth'] = doc.sentences[0].constituency.depth()
                data.append(datum_entry)

                tree = Tree.fromstring(str(doc.sentences[0].constituency))
                annot = doc.text
                list_of_trees.append((tree,annot))
        torch.save(list_of_trees, tree_save_file)

        df = pd.DataFrame(data)
        df.to_csv(save_file, index =  False)

def make_bow_model(save_path = 'bow_model.pt', 
                   rewrite = False, 
                   dataset_csvs = ['dataset_spokencoco_val.csv', 
                                   'dataset_librispeech_train-clean-100.csv']):
    '''
    save_path: path to save the bow model in
    rewrite: if replace the existing bow model or not
    dataset_csvs: csvs generated with preprocessing.py with annotation column
    '''
    if os.path.isfile(save_path) and not rewrite:
        print(f'{save_path} exists already! not overwriting')
        cv = torch.load(save_path)
    else:
        list_of_sents = []
        for dataset in dataset_csvs:
            list_of_sents += list(pd.read_csv(dataset).annot)
        chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"\(\)\[\]\'\xa0]'
        def remove_special_characters(sent):
            sent = re.sub(chars_to_ignore_regex, '', sent).lower()
            return sent
        list_of_sents = list(map(remove_special_characters,list_of_sents))
        list_of_sents = [item.strip() for item in list_of_sents if item.replace(" ",'').isalpha()]
        unique_words = set(' '.join(list_of_sents).split())
        print(f"there are {len(unique_words)} unique words for the bag of words model")        
        cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        cv.fit(list_of_sents)
        torch.save(cv, save_path)
    return cv


def main():

    # spokencoco pre-preprocessing 
    spokencoco_root = '~/SpokenCOCO/'
    spokencoco_split = 'val'
    spokencoco_csv = 'spokencoco_'+spokencoco_split+'.csv'

    if os.path.isfile(spokencoco_csv) == False:
        # making sure the csv file is there, otherwise create the csv file
        spokencoco_root = os.path.expanduser(spokencoco_root)
        json_files = [x for x in os.listdir(spokencoco_root) if 'json' in x]
        split_json = [x for x in json_files if spokencoco_split in x][0]
        json_path = os.path.join(spokencoco_root,split_json)
        print(f"{spokencoco_csv} not found, creating from {json_path}")
        spokencoco_df = read_json_save_csv(json_path)
        spokencoco_df.to_csv(spokencoco_csv, header=None, index = None)
    else:
        print(f"{spokencoco_csv} exists already! not overwriting")
    
    # librispeech pre-preprocessing 
    librispeech_root = '~/work_dir/librispeech-train/'
    libri_split = 'train-clean-100' # 'test-clean'
    librispeech_csv = 'librispeech_'+libri_split+'.csv'

    if os.path.isfile(librispeech_csv) == False:
        # making sure the csv file is there, otherwise create the csv file
        librispeech_root = os.path.expanduser(librispeech_root)
        print(f"{librispeech_csv} not found, creating from {os.path.join(librispeech_root, libri_split)}")
        librispeech_dataset_df = walk_librispeech_dirs(librispeech_root=librispeech_root, libri_split=libri_split)
        librispeech_dataset_df.to_csv(librispeech_csv, header = None, index=False)
    else:
        print(f"{librispeech_csv} exists already! not overwriting")


    csvs = {'scc':'spokencoco_val.csv', 'libri':'librispeech_train-clean-100.csv'}
    for x in csvs:
        print(f"saving {x} as a dataset in csv format")
        dataset_root_dir = os.path.join(librispeech_root, libri_split) if x == 'libri' else spokencoco_root
        make_dataset_csv(csv_file=csvs[x], dataset_root_dir=dataset_root_dir, rewrite=False)
    make_bow_model()

if __name__ == "__main__":
    main()