# Wave to Syntax: Probing spoken language models for syntax

This repo hosts the code for 

## Installation

Clone repo and set up and activate a virtual environment with conda:
```
conda create --name wav2syn --file requirements.txt
conda activate wav2syn
```
The exact configuration of the conda environment used to conduct the experiments can be found in `spec-file.txt`

## Pre-processing 

modify the dataset paths within and run `python preprocessing.py` to generate a dataset csv file for the textCorpus and Corpus classes.

`preprocessing.py` also uses the stanza parser to save all the constituency trees for each utterance in the dataset.

### Datasets

This project have implemented the embedding extraction script for LibriSpeech and SpokenCOCO corpus. You can download the two corpora from the links here [SpokenCOCO](https://data.csail.mit.edu/placesaudio/SpokenCOCO.tar.gz) [LibriSpeech](https://www.openslr.org/12). 

After downloading and extracting the datasets, read the `main()` function in `preprocessing.py` and change the root directories of the datasets and splits you want to use. 

Running `preprocessing.py` generates dataset csv files that can be understood by the probing scripts.

The preprocessing script also makes the bag-of-words model at the same time.

### Models

This repo uses Huggingface Hub to load models.

If you would like to replicate findings with the FaST-VGS family of models, please check instructions on https://github.com/jasonppy/FaST-VGS-Family.

### Extract embeddings from spoken language model

Feature extraction have been implemented in `embedding_generation.py`. It might be desirable to modify `embedding_generation.py ` to limit what model you want to investigate on. The script will save the extracted features including the meanpooled layerwise embedding, the treedepth, the annotation, the audio path, audio length and wordcount to a .pt file under `embeddings`.



## Running TreeDepth probe


Run  
```
python treedepthprobe.py >> treedepth.out
```

## Running TreeKernel probe
Run  
```
python treekernel_prep.py
python treekernelprobe.py >> treekernel.out
```
