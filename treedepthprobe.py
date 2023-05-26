import argparse
import os
from itertools import islice

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, RidgeCV, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm.auto import tqdm
from sklearn.preprocessing import Normalizer as Norm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_regression_model(model_name):
    if model_name == 'logreg':
        model = LogisticRegression(solver = 'saga',multi_class='auto', max_iter=100)
    elif model_name == 'ridge':
        model = RidgeCV(alphas = [ 10**n for n in range(-3, 2) ], cv=10)
    elif model_name == "svm":
        model = SVC(kernel = 'linear', C = 1,max_iter=100)
    return model


def model_fitting(X,y, model):
    if X[0].size == 1:
        X_train, X_test, y_train, y_test = train_test_split(np.array(X).reshape(-1,1), y, random_state = 42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

    model.fit(X_train, y_train)
    if not str(type(model)) == "<class 'sklearn.linear_model._ridge.RidgeCV'>":
        y_pred = model.predict(X_test)
        r2score = r2_score(y_test,y_pred)
        mse = mean_squared_error(y_test,y_pred)
        acc = accuracy_score(y_test,y_pred)
        return r2score, mse, acc
    elif str(type(model)) == "<class 'sklearn.linear_model._ridge.RidgeCV'>":
        model_alpha = model.alpha_
        r2score = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test,y_pred)
        return {
            'r2score': r2score,
            'mse': mse,
            'model_alpha':model_alpha
        }   


def run_baseline(embedding_files = ['embeddings/wav2vec2-base_librispeech_train_extracted.pt',
                                    'embeddings/wav2vec2-base_spokencoco_val_extracted.pt'], 
                                    reg_model_name = 'ridge'):
    embedding_files = [x for x in embedding_files if 'wav2vec2-base' in x]
    reg_model = load_regression_model(reg_model_name)
    for embedding_file in tqdm(embedding_files):
        datasetname = embedding_file.split('_',1)[1].replace('_extracted.pt','')
        _, labels, _ , _, wordcount, audiolen  = torch.load(embedding_file)
        labels = torch.tensor(labels).numpy()
        wordcount = torch.tensor(wordcount).numpy()
        audiolen = torch.tensor(audiolen).numpy()
        lookup_table = {'wordcount': wordcount,
                        'audiolen': audiolen}
        for feat_name in lookup_table.keys():
            results = model_fitting(X = lookup_table[feat_name], y = labels, model=reg_model)
            meta = {
                'modelname': 'baseline',
                'datasetname': datasetname,
                'layer': None,
                'feature': feat_name
            }
            print(meta|results)



def run_model(embedding_files = ['embeddings/wav2vec2-base_librispeech_train_extracted.pt',
                                    'embeddings/wav2vec2-base_spokencoco_val_extracted.pt'], 
              reg_model_name = 'ridge', 
              combi = False,
              with_bow = False,
              bow_embedding_files = None):
    '''
        embedding_files: a list of embedding files to run experiment on
        reg_model_name: default to RidgeCV from sklearn
        combi: if train reg model with combined features
        with_bow: if train reg model with features combined with BoW representations
    '''
    

    reg_model = load_regression_model(reg_model_name)
    for embedding_file in tqdm(embedding_files):
        tqdm.write(f'loading embeddings from {embedding_file}')
        modelname = os.path.basename(embedding_file).split('_',1)[0]
        datasetname = embedding_file.split('_',1)[1].replace('_extracted.pt','')
        embeddings, labels, _,_,wordcount,audiolen  = torch.load(embedding_file)
        embeddings = np.array(embeddings)
        labels = torch.tensor(labels).numpy()
        
        if len(wordcount) == len(labels) and combi == True:
            wordcount = torch.tensor(wordcount).numpy()
            audiolen = torch.tensor(audiolen).numpy()
            lookup_table = {'wordcount': wordcount,
                            'audiolen': audiolen,
                            'wordcount+audiolen': np.column_stack((wordcount,audiolen))}
        elif len(wordcount) != len(labels) and combi == True:
            tqdm.write(f'{embedding_file} does not have combination features! skipping to the next one')
            continue
            
        num_layers = embeddings.shape[1]
        for layer in range(num_layers):
            layer_embs = embeddings[:,layer,:]
            if combi == False:
                results = model_fitting(X = layer_embs, y = labels, model=reg_model)
                meta = {
                    'modelname':modelname,
                    'datasetname': datasetname,
                    'layer': layer,
                    'feature': 'EMB'
                }
                print(meta|results)
            elif combi == True:
                for feat_name in lookup_table.keys():
                    results = model_fitting(X = np.column_stack((layer_embs, lookup_table[feat_name])), y = labels, model=reg_model)
                    meta = {
                        'modelname': modelname,
                        'datasetname': datasetname,
                        'layer': layer,
                        'feature': 'EMB+' + feat_name
                    }
                    print(meta|results)

                
def run_model_with_bow(embedding_files = ['embeddings/wav2vec2-base_librispeech_train_extracted.pt',
                                    'embeddings/wav2vec2-base_spokencoco_val_extracted.pt'], 
              reg_model_name = 'ridge', 
              bow_embedding_files = ['embeddings/BOW_librispeech_train_extracted.pt',
                                     'embeddings/BOW_spokencoco_val_extracted.pt']):
    '''
        embedding_files: a list of embedding files to run experiment on
        reg_model_name: default to RidgeCV from sklearn
        bow_embedding_files = a list of BoW embedding files
    '''
    

    reg_model = load_regression_model(reg_model_name)
    for embedding_file in tqdm(embedding_files):
        tqdm.write(f'loading embeddings from {embedding_file}')
        modelname = os.path.basename(embedding_file).split('_',1)[0]
        datasetname = embedding_file.split('_',1)[1].replace('_extracted.pt','')
        embeddings, labels, _,_,_,_  = torch.load(embedding_file)
        embeddings = np.array(embeddings)
        labels = torch.tensor(labels).numpy()
        
        
        bow_file = [x for x in bow_embedding_files if datasetname in x][0]
        bow_embeddings, _,_,_,_,_  = torch.load(bow_file)
        bow_embeddings = np.squeeze(bow_embeddings)


        num_layers = embeddings.shape[1]
        for layer in range(num_layers):
            layer_embs = embeddings[:,layer,:]

            if bow_embeddings.shape[0] == layer_embs.shape[0]:
                results = model_fitting(X = np.column_stack((layer_embs,bow_embeddings)), y = labels, model=reg_model)
                meta = {
                    'modelname':modelname,
                    'datasetname': datasetname,
                    'layer': layer,
                    'feature': 'EMB+BOW'
                }
                print(meta|results)         


if __name__ == "__main__":
    embedding_path = 'embeddings'
    all_embedding_files = [x for x in os.listdir(embedding_path) if '.pt' in x]
    embedding_files = [os.path.join(embedding_path,x) for x in all_embedding_files if "BOW" not in x]
    bow_embedding_files = [os.path.join(embedding_path,x) for x in all_embedding_files if "BOW" in x]
    run_baseline(embedding_files=embedding_files)
    run_model(embedding_files=embedding_files, combi=False)
    run_model(embedding_files=embedding_files, combi=True)

    # # Due to the large size of the BoW representation, it will take approx. 1hr to run for one dataset.
    # run_model(bow_embedding_files, combi = False)
    

    ## Run model with embedding concatenated with BoW representation is very 
    ## time consuming, be cautious
    # run_model_with_bow()


