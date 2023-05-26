import os
import pandas as pd
import textgrid
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def split_libri_transcription(txt_file):
    with open(txt_file) as file:
        lines = file.readlines()
        for line in lines:
            filename = line.split(' ')[0]+'.txt'
            absolute_path = os.path.join(os.path.dirname(txt_file),filename)
            transcription = ' '.join(line.split(' ')[1:])
            try:
                with open(absolute_path, 'w') as new_file:
                    new_file.write(transcription)
            except:
                print('Error occured')

def do_libri_prep(libri_file_path = "/home/gshen/work_dir/librispeech-train/train-clean-100/"):
    for root, dirs, files in os.walk(libri_file_path):
        # walks the librispeech directory and splits the transcription files (.txt)
        list_of_txts = [x for x in files if 'txt' in x]
        if len(list_of_txts) != 0:
            txt_file = os.path.join(root,list_of_txts[0])
            split_libri_transcription(txt_file)


def write_scc_individual_txt(absolute_path, transcription):
    with open(absolute_path, 'w') as new_file:
        new_file.write(transcription)

def do_scc_prep(spokencoco_csv = 'spokencoco_val.csv', spokencoco_path = '/home/gshen/SpokenCOCO/'):

    df_scc = pd.read_csv(spokencoco_csv, header = None, names = ['path', 'transcription'])
    df_scc['path'] = df_scc['path'].apply(lambda x: os.path.join(spokencoco_path,x))
    df_scc['txtfilename'] = df_scc.path.apply(lambda x: x[:-4])+'.txt'
    list_of_scc_files = list(zip(df_scc['txtfilename'], df_scc['transcription']))
    list_of_scc_files[0]
    for audioseg in list_of_scc_files:
        i, j = audioseg
        write_scc_individual_txt(i,j)

def main_prep():
    do_libri_prep()
    do_scc_prep()


def convert_tg_to_csv(tg_file):    
    # Read a TextGrid object from a file.
    tg = textgrid.TextGrid.fromFile(tg_file)
    list_of_aligned_words = []
    for x in tg[0]:
        if x.mark == "":
            word = "[sil]"
        else:
            word = x.mark
        list_of_aligned_words.append((x.minTime, x.maxTime, word))
        
    df = pd.DataFrame(list_of_aligned_words, columns = ['startTime', 'endTime', 'transcription'])
    csv_name = os.path.basename(tg_file).split('.')[0]+'.csv'

    df.to_csv(os.path.join(os.path.dirname(tg_file),csv_name))

def convert_tg_from_dir_to_csv(tg_path = '/home/gshen/SpokenCOCO/aligned_val/'):
    for root, dirs, files in os.walk(tg_path):
        list_of_tgs = [x for x in files if 'TextGrid' in x]
        if len(list_of_tgs) != 0:
            absolute_path_list_of_tgs = [os.path.join(root,x) for x in list_of_tgs]
            for x in absolute_path_list_of_tgs:
                convert_tg_to_csv(x)