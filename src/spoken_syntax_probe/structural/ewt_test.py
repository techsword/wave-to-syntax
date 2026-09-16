

import json
import os
import random
import pandas as pd
import numpy as np
import torch
import ursa.util as U
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from torchmetrics.functional import pairwise_cosine_similarity
from tqdm.auto import tqdm
from ursa.kernel import Kernel

device = 'cuda' if torch.cuda.is_available() else 'cpu'
import ursa.util as U
from nltk.tree import Tree

import json
import random
import sys
from nltk.tree import Tree

def ewt_json_all():
    import conllu as U

    def id2path(sentid, prefix="ewt_data/"):
        cols = sentid.split('-')
        return (prefix + cols[0] + "/penntree/" + '-'.join(cols[1:-1]) + ".xml.tree", int(cols[-1])-1)
    def get_tree(sentid):
        path, index = id2path(sentid)
        return [Tree.fromstring(line) for line in open(path) ][index]

    def gen_dict_from_conllu(data):
        container_list = []
        for datum in data:
            try:
                container_list.append(dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata)['sent_id']))) 
            except:
                continue
        return container_list
    test =  U.parse(open("UD_English-EWT/en_ewt-ud-dev.conllu").read())
    train = U.parse(open("UD_English-EWT/en_ewt-ud-train.conllu").read())
    # dev = U.parse(open('UD_English-EWT/en_ewt-ud-test.conllu').read())
    # ref = random.sample(train, len(test)//10)
    ref = random.sample(test, 200)
    data_train = gen_dict_from_conllu(train)

    

    data_train = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in train[:10000] ]
    data_ref  = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in ref ]    
    data_dev  = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in dev ]    
    json.dump(dict(ref=data_ref, test=data_train), open("ewt_train_all.json","w"))



# def id2path(sentid, prefix="ewt_data/"):
#     cols = sentid.split('-')
#     return (prefix + cols[0] + "/penntree/" + '-'.join(cols[1:-1]) + ".xml.tree", int(cols[-1])-1)
# def get_tree(sentid):
#     path, index = id2path(sentid)
#     return [Tree.fromstring(line) for line in open(path) ][index]
# test =  U.parse(open("UD_English-EWT/en_ewt-ud-dev.conllu").read())
# train = U.parse(open("UD_English-EWT/en_ewt-ud-train.conllu").read())
# dev = U.parse(open('UD_English-EWT/en_ewt-ud-test.conllu').read())
# data_test = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in test ]
# data_ref  = [ dict(sent=datum.metadata['text'], sentid=datum.metadata['sent_id'], tree=str(get_tree(datum.metadata['sent_id']))) for datum in ref ]    

def compute_kernel_(f, tree1, trees_filtered, normalize = True):
    tree1 = delex(tree1)
    tree1_kern = f(tree1, tree1)
    kernel_container = []
    for data2 in trees_filtered:
        tree2 = delex(data2)
        denom = (tree1_kern * f(tree2, tree2))**0.5 if normalize else 1.0
        kernel_container.append(f(tree1, tree2)/denom)
    return np.array(kernel_container)
def delex(n, leaf="X"):
    if isinstance(n, str): 
        return leaf 
    else: 
        return Tree(n.label(), [ delex(c) for c in n[:] ])
    

def generate_kernel_ewt(ref_pts, test_pts, alpha = 0.5, save_path = 'regress-data', save_name = "ewt_kernel.pt", normalization = True, parallel = False, rewrite = False):
    K = Kernel(alpha=alpha)
    save_file = os.path.join(save_path, save_name)
    if os.path.isfile(save_file) and not rewrite:
        print(f"{save_file} exists already, skipping!")
        tree_kernel_container = torch.load(save_file)
    else:
        if parallel == True:
            from joblib import Parallel, delayed
            tree_kernel_container = Parallel(
                    n_jobs=-1, backend='loky'
                    )(delayed(compute_kernel_)(K,i,ref_pts,normalization) for i in tqdm(test_pts))
            
        elif parallel == False:
            # raise NotImplementedError('non-parallel kernel generation not implemented yet')
            tree_kernel_container = []
            for test_pt in tqdm(test_pts):
                tree_kernel_container.append(compute_kernel_(K,test_pt, ref_pts, normalization))
            
        # return tree_kernel_container
        torch.save(tree_kernel_container, save_file)
    return tree_kernel_container
    

def compare_trees(ewt_entry):
    sent = ewt_entry['sent']
    og_tree = Tree.fromstring(ewt_entry['tree'])
    stanza_tree = Tree.fromstring(str(nlp(sent).sentences[0].constituency))
    print(sent)
    return og_tree, stanza_tree

def stanza_ewt_trees(sents):
    import stanza
    from nltk import Tree
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    list_of_trees = []
    for annot in tqdm(sents):
        doc = nlp(annot)
        tree = Tree.fromstring(str(doc.sentences[0].constituency))
        list_of_trees.append(tree)

    return list_of_trees

# stanza_tree_ref = stanza_ewt_trees(sent_ref)
# stanza_tree_test = stanza_ewt_trees(sent_test)
# generate_kernel_ewt(stanza_tree_ref, stanza_tree_test, save_name='ewt_test_stanza_tk.pt', parallel = True)



def load_model(modelname):
    from transformers import AutoModel, AutoTokenizer
    if modelname == 'bert':
        MODEL_ID = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).to(device) 
    elif modelname == 'bert-large':
        MODEL_ID = "bert-large-uncased"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(MODEL_ID).to(device) 
    return model, tokenizer, MODEL_ID.split("/")[-1]

def sent_emb(sents, tokenizer, model, CLS = True):
    feat_list, annot_list = [],[]
    for annot in tqdm(sents):
        annot_list.append(annot)
        with torch.inference_mode():
            inputs = tokenizer(annot.capitalize(), return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states = True)
            features = outputs.hidden_states
            if CLS == True:
                features = torch.stack(features).squeeze(1)[:,0].detach().cpu().numpy()
            else:                
                features = torch.stack(features).squeeze(1).mean(1).detach().cpu().numpy()
            feat_list.append(features)
    return np.stack(feat_list)
        
def ewt_test(json_file = 'ewt.json',kernel_path = 'ewt_test_data/ewt_original_tk.pt', model_ID = 'bert-large'):
    f = open(json_file)
    ewt = json.load(f)
    sent_ref = [s['sent'] for s in ewt['ref']]
    sent_test = [s['sent'] for s in ewt['test']]
    tree_ref = [Tree.fromstring(s['tree']) for s in ewt['ref']]
    tree_test = [Tree.fromstring(s['tree']) for s in ewt['test']]

    embs_file = model_ID + '_'+ json_file.split('.')[0]+'_embs.pt'
    if os.path.isfile(embs_file):
        embs = torch.load(embs_file)
    else:
        model, tokenizer, MODEL_ID = load_model(model_ID)
        embs = {'ref':sent_emb(sent_ref,tokenizer, model), 
            'test':sent_emb(sent_test,tokenizer, model)}
        torch.save(embs, embs_file)
    
    # emb_test = sent_emb(sent_test, tokenizer, model)
    # emb_ref = sent_emb(sent_ref, tokenizer, model)
    emb_ref = embs['ref']
    emb_test = embs['test']
    if os.path.isfile(kernel_path):

        tk = np.stack(torch.load(kernel_path))
    else:
        filename = json_file.split('.')[0]+'tk.pt'
        tk_raw = generate_kernel_ewt(tree_ref, tree_test, save_name= filename)
        tk = np.stack(tk_raw)
    for layer in range(emb_ref.shape[1]):
        R = Regress()
        test = torch.tensor(emb_test[:,layer,:], dtype = float).to(device)
        ref = torch.tensor(emb_ref[:,layer,:], dtype = float).to(device)
        pd = pairwise_cosine_similarity(test,ref).detach().cpu().numpy()
        pd = np.nan_to_num(pd, copy=True, nan=1.0, posinf=None, neginf=None)
        score = R.fit_report(X=pd, Y = tk)

        result = {
                    'modelname': model_ID,
                    'datasetname': 'EWT',
                    'layer': layer,
                    'kernel': os.path.basename(kernel_path).split('_')[1]
                }
        print(result|score)


def pearson_r_score(Y_true, Y_pred): 
     r =  U.pearson_r(Y_true, Y_pred, axis=0).mean() 
     return r

class Regress:

    default_alphas = [ 10**n for n in range(-3, 2) ]
    metrics = dict(mse       = make_scorer(mean_squared_error, greater_is_better=False),
                   r_sq        = make_scorer(r2_score, greater_is_better=True),
                   pearson_r = make_scorer(pearson_r_score, greater_is_better=True))
                   
    
    def __init__(self, cv=10, alphas=default_alphas):
        self.cv = cv
        self.grid =  {'alpha': alphas }
        self._model = GridSearchCV(Ridge(), self.grid, scoring=self.metrics, cv=self.cv, return_train_score=False, refit=False)

    def fit(self, X, Y):
        self._model.fit(X, Y)
        result = { name: {} for name in self.metrics.keys() }
        for name, scorer in self.metrics.items():
            mean = self._model.cv_results_["mean_test_{}".format(name)] 
            std  = self._model.cv_results_["std_test_{}".format(name)]
            best = mean.argmax()
            result[name]['mean'] = mean[best] * scorer._sign
            result[name]['std']  = std[best]
            result[name]['alpha'] = self.grid['alpha'][best]
        self._report = result

    def fit_report(self, X, Y):
        self.fit(X, Y)
        return self.report()

    def report(self):
        return self._report


def plot(plotting_df, title = 'TreeKernel Task Results'):
    import plotnine as p9
    
    figure = (p9.ggplot(plotting_df,p9.aes('norm_layer', 'pearson_r_mean', color = 'dataset'))
        + p9.geom_point() 
        # + p9.scale_color_manual(colors)
        + p9.geom_line()
        + p9.theme_linedraw()
        + p9.theme(figure_size=(6, 5), dpi=300) 
        # + p9.ylim(0.4,0.7)
        + p9.xlab("Transformer Layer from shallow to deep")
        + p9.ylab("Pearson's r")
        + p9.facet_wrap('~ kernel',ncol=1)
        + p9.ggtitle(title)
        + p9.theme(axis_text_x = p9.element_blank())
        )
    return figure
def read_out_file(outfile, columns, 
                  remove_list = ['modelname', 'datasetname','mse', 'mean','std','alpha','pearson_r','r_sq','layer', 'r2score', 'feature', 'model_', 'kernel']):
    df = pd.read_csv(outfile, names = columns)
    for column in columns:
        for word in remove_list:
            df[column] = df[column].str.replace(word,'')
    df = df.replace('[\(\'\)\{\}\"\s\:\]\[]', '', regex=True)
    df = df.replace('None', 0)
    try:
        df[df.columns[2:]] = df.iloc[:, 2:].apply(pd.to_numeric)
    except:
        df[df.columns[2]] = df.iloc[:, 2].apply(pd.to_numeric)
        df[df.columns[4:]] = df.iloc[:, 4:].apply(pd.to_numeric)

    df['Mode'] = np.where(df['model'].str.contains('fast'), "VGS", 'non-VGS')
    df.loc[df['model'].str.contains('bert|BOW'),'Mode'] = "Text"
    df.loc[df['model'].str.contains('hubert'),'Mode'] = 'non-VGS'

    df['layer'] = df['layer']
    df['norm_layer'] = df.groupby('model').layer.transform(lambda x: x / x.max())
    df = df.sort_values(by=['Mode','model','dataset','layer'])
    df = df.reset_index(drop=True)

    return df

if __name__ == "__main__":
    ewt_test(kernel_path='regress-data/ewt_stanza_kernel.pt')    
    # ewt_test(model_ID='bert')    

    tk_columns = ['model',
                    'dataset',
                    'layer',
                    'kernel',
                    'mse_mean',
                    'mse_std',
                    'mse_alpha',
                    'r_sq_mean',
                    'r_sq_std',
                    'r_sq_alpha',
                    'pearson_r_mean',
                    'pearson_r_std',
                    'pearson_r_alpha']

    df = read_out_file('ewt_results.txt', columns = tk_columns)
    plot(df).save('ewt_test.png')

    df_ = read_out_file('results/tk_bert.out', columns = ['model',
                    'dataset',
                    'layer',
                    'mse_mean',
                    'mse_std',
                    'mse_alpha',
                    'r_sq_mean',
                    'r_sq_std',
                    'r_sq_alpha',
                    'pearson_r_mean',
                    'pearson_r_std',
                    'pearson_r_alpha'])
    df_['kernel'] = df_['dataset'].str.replace('_extracted_CLS.pt','')
    exp_df = df_[df_['dataset'].str.contains('spoken')&df_['model'].str.contains('large')&(df_.layer!=0)]
    # exp_df.loc[:,'kernel'] = exp_df['dataset'].str.replace('_extracted_CLS.pt', '')
    merged = pd.concat([df, exp_df])
    plot(merged)

'''
Test with more data from EWT?
https://universaldependencies.org/treebanks/en_ewt/index.html

write a document summarize what happened here: e.g. what could have caused the difference 
'''