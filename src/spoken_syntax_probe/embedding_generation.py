# GPU selection is left to the environment instead of being hardcoded here.
# To use a specific GPU, export it in your shell before launching, e.g.:
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python embedding_generation.py

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoModel, AutoTokenizer, Wav2Vec2Config)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from .utils.custom_classes import Corpus, textCorpus


def load_fast_vgs_model(model_path):
    '''
    instructions on https://github.com/jasonppy/FaST-VGS-Family
    '''
    from fast_vgs_family.models import w2v2_model

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

def select_vgs_model(modelname, fast_vgs_root='fast_vgs_family/model_path'):
    '''
    see load_fast_vgs_model() and select_model()

    fast_vgs_root: directory that contains the manually downloaded FaST-VGS
    checkpoints (fast-vgs-coco/ and fast-vgs-plus-coco/). See README "Models".
    '''
    # Loading VGS models
    if modelname == "fast-vgs":
        checkpoint_id = os.path.join(fast_vgs_root, 'fast-vgs-coco')
        MODEL_ID = "fast-vgs"
        model = load_fast_vgs_model(checkpoint_id)
    elif modelname == "fast-vgs-plus":
        checkpoint_id = os.path.join(fast_vgs_root, 'fast-vgs-plus-coco')
        MODEL_ID = "fast-vgs-plus"
        model = load_fast_vgs_model(checkpoint_id)
    else:
        raise NotImplementedError(f"loading {modelname} is not implemented")
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


def run_feat_gen(modelname='wav2vec2_small', dataset_csv="dataset_spokencoco_val.csv",
                 save_dir='embeddings', rewrite=False, CLS=False,
                 fast_vgs_root='fast_vgs_family/model_path'):
    '''
    modelname: name of the model, string
    dataset_csv: relative path of the dataset csv to pass on to textCorpus or Corpus classes
    save_dir: filepath for saving generated embeddings
    rewrite: assign to yes if rewrite is intended
    CLS: only concerns the BERT and DeBERTa models, choose if the embedding is a meanpooled vector or just the CLS vector
    fast_vgs_root: directory with the manually downloaded FaST-VGS checkpoints
    '''
    if 'fast-vgs' not in modelname:
            model, tokenizer, model_ID = select_model(modelname=modelname)
    else:
        try:
            model, tokenizer, model_ID = select_vgs_model(modelname=modelname,
                                                          fast_vgs_root=fast_vgs_root)
        except Exception as err:
            raise LookupError(
                f"loading {modelname} failed; check --fast_vgs_root and the "
                f"FaST-VGS setup in the README") from err

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
        # Protocol 5 legacy serialization: load-identical format, lower peak
        # memory on the many-small-array feature lists (see rsa.py).
        torch.save([feat_list, lab_list,annot_list,wav_path_list, wordcount_list, audiolen_list], os.path.join(save_dir,save_file),
                   pickle_protocol=5, _use_new_zipfile_serialization=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate layer-wise embeddings for a list of models and datasets.')
    parser.add_argument('--models', nargs='+',
                        default=['BOW', 'wav2vec2_small'],
                        help='Model names to run (keys of models_dict in select_model, '
                             'or fast-vgs / fast-vgs-plus).')
    parser.add_argument('--datasets', nargs='+',
                        default=['dataset_librispeech_train-clean-100.csv',
                                 'dataset_spokencoco_val.csv'],
                        help='Dataset csv files (generated by preprocessing.py).')
    parser.add_argument('--save_dir', default='embeddings',
                        help='Directory to save the extracted .pt files.')
    parser.add_argument('--fast_vgs_root', default='fast_vgs_family/model_path',
                        help='Directory containing the manually downloaded '
                             'fast-vgs-coco/ and fast-vgs-plus-coco/ checkpoints '
                             '(see README "Models").')
    parser.add_argument('--rewrite', action='store_true',
                        help='Overwrite existing embedding files.')
    parser.add_argument('--no_cls', action='store_true',
                        help='Use mean-pooled text features instead of the CLS vector.')
    cli = parser.parse_args()

    # Iterates through the models and datasets to generate embeddings
    for modelname in tqdm(cli.models):
        for dataset_csv in tqdm(cli.datasets):
            run_feat_gen(modelname, dataset_csv, save_dir=cli.save_dir,
                         rewrite=cli.rewrite, CLS=not cli.no_cls,
                         fast_vgs_root=cli.fast_vgs_root)