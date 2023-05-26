import json
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from tqdm.auto import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


sr = 16000


def loading_fairseq_model(model_file):
    import fairseq

    from torchaudio.models.wav2vec2.utils import import_fairseq_model
    model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_file])
    original = model[0]
    imported = import_fairseq_model(original)
    return imported

def loading_huggingface_model(hf_model):
    from torchaudio.models.wav2vec2.utils import import_huggingface_model
    model = import_huggingface_model(hf_model)
    return model

def walk_librispeech_dirs(librispeech_root, libri_split):
    libri_split_path = os.path.join(librispeech_root,libri_split)
    filelist = []
    for root, dir, file in os.walk(libri_split_path):
        filelist.append(file)
    import itertools
    flatlist_raw = list(itertools.chain(*filelist))
    fileid_pattern = re.compile(r'\d+\-\d+-\d+')
    flatlist = [s for s in flatlist_raw if fileid_pattern.match(s)]
    txtlist = [os.path.join(*[re.split(r'\.|-', x)[0],re.split(r'\.|-', x)[1],x]) for x in flatlist if 'txt' in x]
    flaclist = [os.path.join(*[re.split(r'\.|-', x)[0],re.split(r'\.|-', x)[1],x]) for x in flatlist if 'flac' in x]
    df = pd.DataFrame(flaclist)
    df['fileid'] = df[0].apply(lambda x : os.path.split(x)[1])
    df['fileid'] = df['fileid'].str.replace('.flac', '', regex=True)
    df = df.rename({0:'fullpath'},axis=1)
    txt_out = []
    for txt_file in tqdm(txtlist):
        with open(os.path.join(libri_split_path, txt_file)) as f:
            txt_dict = {}

            sent = f.readline()
            file_id = os.path.split(txt_file)[-1]
            txt_dict['sent'] = sent
            txt_dict['fileid'] = file_id
            txt_out.append(txt_dict)
    df_txt = pd.DataFrame(txt_out)
    df_txt['fileid'] = df_txt['fileid'].str.replace('.txt', '', regex=True)
    merge_df = pd.merge(df,df_txt, on = 'fileid')
    merge_df = merge_df.drop(columns=['fileid'])
    return merge_df
    
def make_bow(doc):
    
    doc = list(map(str.lower, doc))
    unique_words = set(' '.join(doc).split())
    print(f'there are {len(unique_words)} unique words')
    index_dict = {}
    for ind, i in enumerate(sorted(unique_words)):
        index_dict[i] = ind
    cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    count_occurrences = cv.fit_transform(doc)
    return count_occurrences.toarray()

def read_json_save_csv(json_path):
    '''Reading the SpokenCOCO json file and extracting the text and wav file pairs'''
    with open(json_path) as json_file:
        data = json.load(json_file)
    text = []
    wav = []
    for image in data['data']:
        for caption in image['captions']:
            text.append(caption['text'])
            wav.append(caption['wav'])


    '''
    Using pandas to combine the two lists extracted in the cells above and 
    save the resulting dataframe into a csv file
    '''
    dict_spokencoco = dict(zip(text,wav))
    spokencoco_val = pd.Series(dict_spokencoco, name = 'wav')
    spokencoco_val.index.name = 'text'
    spokencoco_val = spokencoco_val.reset_index()
    column_titles = ['wav', 'text']
    spokencoco_val = spokencoco_val.reindex(columns = column_titles)
    return spokencoco_val



def get_weird_sents(corpus_csv, root_dir):
    from custom_classes import Corpus
    corpus_ = Corpus(corpus_csv, root_dir)
    weird_sents = [x for x in [corpus_.get_depth(i) for i in range(len(corpus_))] if x[1] > len(str.split(x[0]))+2]
    print(f'there are total {len(weird_sents)} sentences')
    print(weird_sents)

def generate_tree(dataset, num_entries = None):
    import stanza
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    from nltk.tree import Tree

    text = [x[-2] for x in dataset[:num_entries]]

    tree_list = [str(nlp(sent).sentences[0].constituency) for sent in text]
    nltk_tree_list = [Tree.fromstring(x) for x in tree_list]

    return list(zip((nltk_tree_list,text)))

def recover_from_triu(matrices):

    size_X = matrices[-1]
    X = np.zeros((size_X,size_X))
    full_matrices = []
    for v in matrices[:-1]:
        X[np.triu_indices(X.shape[0], k = 1)] = v
        X = X + X.T
        full_matrices.append(X)
    return full_matrices