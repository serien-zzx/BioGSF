# BioGSF: A Graph-Driven Semantic Feature Integration Framework for Biomedical Relation Extraction

This project is a medical entity relationship extraction model. You can view related articles from the following address: BioGSF: A Graph-Driven Semantic Feature Integration Framework for Biomedical Relation Extraction.

# Overview
The automatic and accurate extraction of diverse biomedical relations from literature constitutes the core elements of medical knowledge graphs, which are indispensable for healthcare artificial intelligence. Currently, fine-tuning through stacking various neural networks on pre-trained language models (PLMs) represents a common framework for end-to-end resolution of the biomedical relation extraction (RE) problem. Nevertheless, sequence-based PLMs, to a certain extent, fail to fully exploit the connections between semantics and the topological features formed by these connections. 

In this study, we presented a graph-driven framework named BioGSF for RE from the literature by integrating shortest dependency paths (SDP) with entity-pair graph through the employment of the graph neural network model. Initially, we leveraged dependency relationships to obtain the SDP between entities and incorporated this information into the entity-pair graph. Subsequently, the graph attention network was utilized to acquire the topological information of the entity-pair graph. Ultimately, the obtained topological information was combined with the semantic features of the contextual information for relation classification.


# Model
Our model BioGSF is primarily divided into three modules: the embedding layer, the feature fusion layer, and the classification layer. 

![Model](https://github.com/serien-zzx/BioGSF/blob/main/Figure/Model.png)

# Dataset
Due to copyright related issues, you can get from [MKG-GC](https://github.com/KeDaCoYa/MKG-GC?tab=readme-ov-file#requirements) and [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/).



# Installtion
Our model is developed based on Pytorch and you need to install the following relevant dependencies.
```
transformers==3.4.0
python>=3.8
scispacy==0.5.3
pytorch >= 1.7
```
Alternatively, you can use the `pip` command to install the environment.
```
pip -r requirements.txt
```
Now that you have configured the relevant environment, you still need to go to [ScispaCy](https://github.com/allenai/scispacy) and download the relevant files, which we use `en_core_sci_lg`.

# Data preparation
You can go to [here](https://github.com/serien-zzx/BioGSF/tree/main/re/original_dataset/Your_Dataset) to complete the relevant data preparation and we have put the relevant instructions there.

# Training

We provide the contents of the training script file as follows, through which you can perform the relevant experiments：
```
BERT_DIR='BERT_DIR'
DATA_SET='Your DATA SET' 
NEG_LEN=64

python ./re_main.py  \
             --num_epochs=15 \
             --gpu_id='0' \
             --use_gpu=True \
             --use_n_gpu=True \
             --bert_lr=1e-5\
             --use_scheduler=True\
             --fixed_batch_length=True \
             --freeze_bert=True\
             --max_len=512 \
             --neg_len=${NEG_LEN}\
             --batch_size=32 \
             --bert_name='biobert' \
             --model_name='gat_neg_single_entity_marker' \
             --scheme=36 \
             --data_format='neg_single' \
             --dataset_type='original_dataset' \
             --dataset_name=${DATA_SET} \
             --class_type='neg_mid' \
             --run_type='normal' \
             --print_step=1 \
             --bert_dir=${BERT_DIR}\
             --train_verbose=True \
             --save_model=True\
```
>If you need to test your dataset,you need to change `re_main` which I've labelled in the file.
# Predicate
This section has scripts for predictions:
```
BERT_DIR='BERT_DIR'
DATA_SET='Your DATA SET'
NEG_LEN=64 

python ./re_predicate.py  \
              --num_epochs=1 \
              --gpu_id='0' \
              --use_gpu=True \
              --use_n_gpu=True \
              --bert_lr=1e-5\
              --use_scheduler=True\
              --fixed_batch_length=True \
              --freeze_bert=True\
              --max_len=512 \
              --neg_len=${NEG_LEN}\
              --batch_size=32 \
              --bert_name='biobert' \
              --model_name='gat_neg_single_entity_marker' \
              --scheme=36 \
              --data_format='neg_single' \
              --dataset_type='original_dataset' \
              --dataset_name=${DATA_SET} \
              --class_type='neg_mid' \
              --run_type='normal' \
              --print_step=1 \
              --bert_dir=${BERT_DIR}\ 
              --train_verbose=True \
```
