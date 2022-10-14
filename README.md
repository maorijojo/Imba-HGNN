# Imba-HGNN
We provide the code (in pytorch) and datasets for our work: "Imba-HGNN".

## 1. Desription
The repository is organised as follows:

* data/: contains the 3 benchmark datasets: ACM, DBLP and IMDB. All datasets will be processed on the fly. Please extract the file of each dataset before running.

* layers/: contains GNN layers Imba-HGNN needs: fea2embed_layer, fea2mp_layer and gat_layer. 

* ImbaHgnn/: contains ImbaHgnn model we proposed.

* utils/: contains two tool python files.

* params/: stores parameter files which will be generated in training process.

## 2. Requirements
To install required packages
- pip install -r requirements.txt

## 3. Running experiments
To run our model, please run these commands regarding to specific dataset:
- python run.py --dataset acm --sparse
- python run.py --dataset dblp --sparse
- python run.py --dataset imdb --sparse
