# BioGSF: A Graph-Driven Semantic Feature Integration Framework for Biomedical Relation Extraction
# Introduction
This project is a medical entity relationship extraction model. You can view related articles from the following address: BioGSF: A Graph-Driven Semantic Feature Integration Framework for Biomedical Relation Extraction.

# Dataset
Due to copyright related issues, you can get from [MKG-GC](https://github.com/KeDaCoYa/MKG-GC?tab=readme-ov-file#requirements) and [BioRED](https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/).

# Requirements
```
transformers==3.4.0
python>=3.8
scispacy==0.5.3
pytorch >= 1.7
```

# Training
We provide the contents of the training script file as follows, through which you can perform the relevant experiments：
```
BERT_DIR='BERT_DIR'

python ./re_main.py  \
             --num_epochs=15 \
             --gpu_id='0' \
             --use_gpu=True \
             --use_n_gpu=True \
             --bert_lr=1e-5\
             --use_scheduler=True\
             --fixed_batch_length=True \
             --freeze_bert=True\
             --l1_lambda=0\
             --max_len=512 \
             --neg_len=64\
             --batch_size=32 \
             --bert_name='biobert' \
             --model_name='gat_neg_single_entity_marker' \
             --scheme=36 \
             --data_format='neg_single' \
             --dataset_type='original_dataset' \
             --dataset_name='BIORED' \
             --class_type='neg_mid' \
             --run_type='normal' \
             --print_step=1 \
             --bert_dir=${BERT_DIR}\
             --train_verbose=True \
             --save_model=True\
```
# Predicate
This section has scripts for predictions:
```
BERT_DIR='BERT_DIR'

python ./re_predicate.py  \
              --num_epochs=1 \
              --gpu_id='0' \
              --use_gpu=True \
              --use_n_gpu=True \
              --bert_lr=1e-5\
              --use_scheduler=True\
              --fixed_batch_length=True \
              --freeze_bert=True\
              --l1_lambda=0\
              --max_len=512 \
              --neg_len=64\
              --batch_size=32 \
              --bert_name='biobert' \
              --model_name='gat_neg_single_entity_marker' \
              --scheme=36 \
              --data_format='neg_single' \
              --dataset_type='original_dataset' \
              --dataset_name='your predicate dataset' \
              --class_type='neg_mid' \
              --run_type='normal' \
              --print_step=1 \
              --bert_dir=${BERT_DIR}\ 
              --train_verbose=True \
```
