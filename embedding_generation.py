import os
import pickle
from itertools import islice

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer, Wav2Vec2Config,
                          Wav2Vec2Model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from utils.custom_classes import Corpus, textCorpus


def load_fast_vgs_model(model_path):
    '''
    instructions on https://github.com/jasonppy/FaST-VGS-Family
    '''
    from fast_vgs_family.models import fast_vgs, w2v2_model

    # load args
    with open(f"{model_path}/args.pkl", "rb") as f:
        args = pickle.load(f)
    # load weights
    weights = torch.load(os.path.join(model_path, "best_bundle.pth"))
    if 'plus' in model_path:
        args_dict = vars(args)
        args_dict['trim_mask'] = False
    model = w2v2_model.Wav2Vec2Model_cls(args)
    model.carefully_load_state_dict(weights['dual_encoder']) 

    return model

def select_vgs_model(modelname):
    '''
    see load_fast_vgs_model() and select_model()
    '''
    # Loading VGS models
    if modelname == "fast-vgs":
        checkpoint_id = 'fast_vgs_family/model_path/fast-vgs-coco'
        MODEL_ID = "fast-vgs"
        model = load_fast_vgs_model(checkpoint_id)
    elif modelname == "fast-vgs-plus":
        checkpoint_id = 'fast_vgs_family/model_path/fast-vgs-plus-coco'
        MODEL_ID = "fast-vgs-plus"
        model = load_fast_vgs_model(checkpoint_id)
    hf_model = None   
    tokenizer = None
    return model, tokenizer, MODEL_ID.split("/")[-1]

def select_model(modelname):   
    '''
    loads model using huggingface hub or local path, 
    returns (model, tokenizer, model_ID), if the model is not BERT or DeBERTa, tokenizer will be None
    '''
    models_dict = {'hubert_base_ls960':'facebook/hubert-base-ls960',
                   'wav2vec2_small_ft':'techsword/wav2vec2-small-libri-scc-ft-ckp-10000',
                   'wav2vec2_small': 'facebook/wav2vec2-base',
                   'wav2vec2_large_ft':"jonatasgrosman/wav2vec2-large-english",
                   'wav2vec2_large':'facebook/wav2vec2-large',
                   'wav2vec2_random':'wav2vec2-random',
                   'bert':'bert-base-uncased',
                   'bert-large':'bert-large-uncased'}
    text_models = ['bert', 'bert-large']
    
    if modelname in models_dict:
        MODEL_ID = models_dict[modelname]
        if modelname in text_models:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            hf_model = None
            model = AutoModel.from_pretrained(MODEL_ID).to(device)
        elif modelname == 'wav2vec2_random':
            tokenizer = None
            # Initializing a model (with random weights) from the facebook/wav2vec2-base-960h style configuration
            configuration = Wav2Vec2Config()
            model = AutoModel.from_config(configuration)
        else:
            tokenizer = None
            model = AutoModel.from_pretrained(MODEL_ID)
            MODEL_ID = 'wav2vec2-small-ft' if 'checkpoint' in MODEL_ID else MODEL_ID
    elif modelname == 'BOW':
        MODEL_ID = "BOW"
        save_path = 'bow_model.pt'
        model = torch.load(save_path)
        tokenizer = None
    else: 
        raise NotImplementedError(f"loading {modelname} is not implemented")    
    return model, tokenizer, MODEL_ID.split("/")[-1]


def run_feat_gen(modelname = 'wav2vec2_small', dataset_csv = "dataset_spokencoco_val.csv", save_dir = 'embeddings', rewrite = False, CLS = False):
    '''
    modelname: name of the model, string
    dataset_csv: relative path of the dataset csv to pass on to textCorpus or Corpus classes
    save_dir: filepath for saving generated embeddings
    rewrite: assign to yes if rewrite is intended
    CLS: only concerns the BERT and DeBERTa models, choose if the embedding is a meanpooled vector or just the CLS vector
    '''
    if 'fast-vgs' not in modelname:
            model, tokenizer, model_ID = select_model(modelname=modelname)
    else:
        try:
            model, tokenizer, model_ID = select_vgs_model(modelname=modelname)
        except:
            raise LookupError(f"loading {modelname} failed")
        
    dataset_ID = dataset_csv.split(".")[0].split("-")[0].replace("dataset_", "")      
    save_file = "_".join([model_ID,dataset_ID])+'_extracted.pt'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.isfile(os.path.join(save_dir,save_file)) and not rewrite:
        tqdm.write(f"{save_file} exists already! skipping to the next one")
    else:
        tqdm.write(f"{save_file} generating extracted embeddings from {dataset_ID} using {model_ID}")
        if model_ID == "BOW":
            dataset = textCorpus(csv_file = dataset_csv)
            annot_list, lab_list, wav_path_list = zip(*dataset)
            feat_list, audiolen_list, wordcount_list = [], [], []
            feat_list = model.transform(tqdm(annot_list)).toarray()
            feat_list = np.expand_dims(feat_list, axis=1)
        elif tokenizer != None:
            save_file = save_file.replace('.pt', '_CLS.pt') if CLS == True else save_file
            dataset = textCorpus(csv_file = dataset_csv)
            dataloader = DataLoader(dataset, shuffle = False, num_workers=0)
            feat_list, annot_list, lab_list, audiolen_list,wordcount_list, wav_path_list = [],[],[],[],[],[]
            model = model.to(device)
            for annot, depth, audio_name in tqdm(dataloader):
                annot_list.append(annot)
                lab_list.append(depth)
                wav_path_list.append(audio_name)
                with torch.inference_mode():
                    inputs = tokenizer(annot[0].capitalize(), return_tensors="pt").to(device)
                    outputs =model(**inputs, output_hidden_states = True)
                    features = outputs.hidden_states
                    if CLS == True:
                        features = torch.stack(features).squeeze(1)[:,0].detach().cpu().numpy()
                    else:                
                        features = torch.stack(features).squeeze(1).mean(1).detach().cpu().numpy()
                    feat_list.append(features)
            # tqdm.write(f"there are {len(feat_list)} in the extracted dataset, each tensor is {features[0].shape}")
        else:
            dataset = Corpus(csv_file=dataset_csv)
            dataloader = DataLoader(dataset, shuffle = False, num_workers=0)
            feat_list, annot_list, lab_list, audiolen_list,wordcount_list, wav_path_list = [],[],[],[],[],[]
            model = model.to(device)
            for audio, annot, lab, audiolen, wordcount, wav_path in tqdm(dataloader):
                annot_list.append(annot)
                lab_list.append(lab)
                audiolen_list.append(audiolen)
                wordcount_list.append(wordcount)
                wav_path_list.append(wav_path)
                with torch.inference_mode():
                    if 'fast-vgs' not in model_ID:
                        # features, _ = model.to(device).extract_features(audio.squeeze(1).to(device))
                        outputs = model(audio.squeeze(1).to(device), output_hidden_states = True)
                        features = outputs.hidden_states


                    elif 'fast-vgs' in model_ID:
                        
                        features = model(source=audio.squeeze(1).to(device), padding_mask=None, mask=False, superb=True)['hidden_states']
                    features = torch.stack(features).squeeze(1).mean(1).detach().cpu().numpy()
                    feat_list.append(features)
        tqdm.write(f'finished generation and saving features to {save_file}')
        torch.save([feat_list, lab_list,annot_list,wav_path_list, wordcount_list, audiolen_list], os.path.join(save_dir,save_file), pickle_protocol = 4)


if __name__ == "__main__":
    models = ['BOW', 'wav2vec2_small']#,'wav2vec2_random', 'wav2vec2_large_ft', 'wav2vec2_small_ft', 'hubert_base_ls960','bert']#,'deberta','fast-vgs-plus','fast-vgs','bert-large']
    datasets = ['dataset_librispeech_train-clean-100.csv', "dataset_spokencoco_val.csv"]
    # Iterates through the model and datasets to generate embeddings
    for modelname in tqdm(models):
        for dataset_csv in tqdm(datasets):
            run_feat_gen(modelname, dataset_csv, rewrite=False, CLS=True)