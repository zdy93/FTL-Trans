# FTL-Trans
This repo hosts relevant scripts for Flexible Time-Aware LSTM Transformer (**FTL-Trans**). 

## Requirement
### Language
* Python3 == 3.x.x
### Module
* torch==1.3.1+cu92
* pytorch-pretrained-bert==0.6.2
* pytorch-transformers==1.2.0
* tqdm==4.37.0
* dotmap==1.3.8
* six==1.13.0
* matplotlib==3.1.1
* numpy==1.17.3
* pandas==0.25.3
## Dataset
We use [MIMIC-III](https://mimic.physionet.org/gettingstarted/access/). We refer users to the link for requesting access. You can also use some other clinical notes or non-clinical documents as input.

File system expected:
```Linux
data/
  test.csv
  train.csv
  val.csv
```
## Pretrained Model
In our paper, we initialize the transformer layer with [ClinicalBERT](https://github.com/kexinhuang12345/clinicalBERT). We refer user to the link for requesting pre-trained model. You can also use some other pre-trained models, like [BERT](https://github.com/huggingface/transformers).
## Model Prediction
Below list the scripts for running prediction. The file [run_clbert_ftlstm.py](./run_clbert_ftlstm.py) contains the code for the FT-Trans. Other files named as run_\[model\].py contain codes for baseline models.
```cmd
python3 run_clbert_ftlstm.py
  --data_dir ./data
  --train_data train.csv
  --val_data val.csv
  --test_data test.csv
  --log_path ./log.txt
  --bert_model ./pretraining
  --embed_mode all
  --task_name FTL-Trans_Prediction
  --max_seq_length 128
  --train_batch_size 32
  --eval_batch_size 1
  --learning_rate 2e-5
  --num_train_epochs 3
  --warmup_proportion 0.1
  --max_chunk_num 32
  --seed 42
  --gradient_accumulation_steps 1
  --output_dir ./exp_FTL-Trans
  --save_model True
```
We refer users to [run_clbert_ftlstm.py](./run_clbert_ftlstm.py) for detalied explanation of each parameter.

We also provide [preprocessing.py](./preprocessing.py) and [split_into_chunk.py](./split_into_chunk.py) for preprocessing data and spliting data into chunks. However, if your data does not has the same format as ours, which means that your data does not have the columns that we have (Adm_ID, Note_ID, chartdate, charttime, category, TEXT, dischtime, Label). You need to modify the code before implmenting preprocessing. 
