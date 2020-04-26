#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""

from tqdm import tqdm, trange
import pandas as pd
import io
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from pytorch_transformers import BertTokenizer
from other_func import write_log, preprocess1, preprocessing
from sklearn.model_selection import KFold


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data file path."
                             " Should be the .tsv file (or other data file) for the task.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the processed data will be written.")
    parser.add_argument("--temp_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the intermediate processed data will be written.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task.")
    parser.add_argument("--log_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The log file path.")
    parser.add_argument("--id_num",
                        default=None,
                        type=int,
                        required=True,
                        help="The number of admission ids that we want to use for each categories.")
    parser.add_argument("--random_seed",
                        default=1,
                        type=int,
                        required=True,
                        help="The random_seed for train/val/test split.")
    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    ## Other parameters
    parser.add_argument("--Kfold",
                        default=None,
                        type=int,
                        required=False,
                        help="The number of folds that we want ot use for cross validation. "
                             "Default is not doing cross validation")

    args = parser.parse_args()
    RANDOM_SEED = args.random_seed
    LOG_PATH = args.log_path
    TEMP_DIR = args.temp_dir

    if os.path.exists(TEMP_DIR) and os.listdir(TEMP_DIR):
        raise ValueError("Temp Output directory ({}) already exists and is not empty.".format(TEMP_DIR))
    os.makedirs(TEMP_DIR, exist_ok=True)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    original_df = pd.read_csv(args.original_data, header=None)
    original_df.rename(columns={0: "Adm_ID",
                                1: "Note_ID",
                                2: "chartdate",
                                3: "charttime",
                                4: "TEXT",
                                5: "Label"}, inplace=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    write_log(("New Pre-processing Job Start! \n"
              "original_data: {}, output_dir: {}, temp_dir: {} \n"
              "task_name: {}, log_path: {}, id_num: {}\n"
              "random_seed: {}, bert_model: {}").format(args.original_data, args.output_dir, args.temp_dir,
                                                         args.task_name, args.log_path, args.id_num,
                                                         args.random_seed, args.bert_model), LOG_PATH)

    for i in range(int(np.ceil(len(original_df) / 10000))):
        write_log("chunk {} tokenize start!".format(i), LOG_PATH)
        df_chunk = original_df.iloc[i * 10000:(i + 1) * 10000].copy()
        df_processed_chunk = preprocessing(df_chunk, tokenizer)
        df_processed_chunk = df_processed_chunk.astype({'Adm_ID': 'int64', 'Note_ID': 'int64', 'Label': 'int64'})
        temp_file_dir = os.path.join(TEMP_DIR, 'Processed_{}.csv'.format(i))
        df_processed_chunk.to_csv(temp_file_dir, index=False)

    df = pd.DataFrame({'Adm_ID': [], 'Note_ID': [], 'TEXT': [], 'Input_ID': [],
                       'Label': [], 'chartdate': [], 'charttime': []})
    for i in range(int(np.ceil(len(original_df) / 10000))):
        temp_file_dir = os.path.join(TEMP_DIR, 'Processed_{}.csv'.format(i))
        df_chunk = pd.read_csv(temp_file_dir, header=0)
        write_log("chunk {} has {} notes".format(i, len(df_chunk)), LOG_PATH)
        df = df.append(df_chunk, ignore_index=True)

    result = df.Label.value_counts()
    write_log(
        "In the full dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}".format(result[1],
                                                                                          result[0]),
        LOG_PATH)

    dead_ID = pd.Series(df[df.Label == 1].Adm_ID.unique())
    not_dead_ID = pd.Series(df[df.Label == 0].Adm_ID.unique())
    write_log("Total Positive Patients' ids: {}, Total Negative Patients' ids: {}".format(len(dead_ID), len(not_dead_ID)), LOG_PATH)

    not_dead_ID_use = not_dead_ID.sample(n=args.id_num, random_state=RANDOM_SEED)
    dead_ID_use = dead_ID.sample(n=args.id_num, random_state=RANDOM_SEED)

    if args.Kfold is None:
        id_val_test_t = dead_ID_use.sample(frac=0.2, random_state=RANDOM_SEED)
        id_val_test_f = not_dead_ID_use.sample(frac=0.2, random_state=RANDOM_SEED)

        id_train_t = dead_ID_use.drop(id_val_test_t.index)
        id_train_f = not_dead_ID_use.drop(id_val_test_f.index)

        id_val_t = id_val_test_t.sample(frac=0.5, random_state=RANDOM_SEED)
        id_test_t = id_val_test_t.drop(id_val_t.index)
        id_val_f = id_val_test_f.sample(frac=0.5, random_state=RANDOM_SEED)
        id_test_f = id_val_test_f.drop(id_val_f.index)

        id_test = pd.concat([id_test_t, id_test_f])
        test_id_label = pd.DataFrame(data=list(zip(id_test, [1] * len(id_test_t) + [0] * len(id_test_f))),
                                     columns=['id', 'label'])

        id_val = pd.concat([id_val_t, id_val_f])
        val_id_label = pd.DataFrame(data=list(zip(id_val, [1] * len(id_val_t) + [0] * len(id_val_f))),
                                    columns=['id', 'label'])

        id_train = pd.concat([id_train_t, id_train_f])
        train_id_label = pd.DataFrame(data=list(zip(id_train, [1] * len(id_train_t) + [0] * len(id_train_f))),
                                      columns=['id', 'label'])

        mortality_train = df[df.Adm_ID.isin(train_id_label.id)]
        mortality_val = df[df.Adm_ID.isin(val_id_label.id)]
        mortality_test = df[df.Adm_ID.isin(test_id_label.id)]
        mortality_not_use = df[
            (~df.Adm_ID.isin(train_id_label.id)) & (~df.Adm_ID.isin(val_id_label.id) & (~df.Adm_ID.isin(test_id_label.id)))]

        train_result = mortality_train.Label.value_counts()

        val_result = mortality_val.Label.value_counts()

        test_result = mortality_test.Label.value_counts()

        no_result = mortality_not_use.Label.value_counts()

        if len(no_result) == 2:
            write_log(("In the train dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                       "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                       "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                       "In the not use dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}").format(
                train_result[1],
                train_result[0],
                val_result[1],
                val_result[0],
                test_result[1],
                test_result[0],
                no_result[1],
                no_result[0]),
                LOG_PATH)
        else:
            write_log(("In the train dataset Positive Patients' Notes: {}, Negative  Patients' Notes: {}\n"
                       "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                       "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                       "In the not use dataset Negative  Patients' Notes: {}").format(train_result[1],
                                                                                      train_result[0],
                                                                                      val_result[1],
                                                                                      val_result[0],
                                                                                      test_result[1],
                                                                                      test_result[0],
                                                                                      no_result[0]),
                      LOG_PATH)

        mortality_train.to_csv(os.path.join(args.output_dir, 'train.csv'), index=False)
        mortality_val.to_csv(os.path.join(args.output_dir, 'val.csv'), index=False)
        mortality_test.to_csv(os.path.join(args.output_dir, 'test.csv'), index=False)
        mortality_not_use.to_csv(os.path.join(args.output_dir, 'not_use.csv'), index=False)
        df.to_csv(os.path.join(args.output_dir, 'full.csv'), index=False)

        write_log("Data saved in the {}".format(args.output_dir), LOG_PATH)
    else:
        folds_t = KFold(args.Kfold, False, RANDOM_SEED)
        folds_f = KFold(args.Kfold, False, RANDOM_SEED)
        dead_ID_use.reset_index(inplace=True, drop=True)
        not_dead_ID_use.reset_index(inplace=True, drop=True)
        for num, ((train_t, test_t), (train_f, test_f)) in enumerate(zip(folds_t.split(dead_ID_use),
                                                                         folds_f.split(not_dead_ID_use))):
            id_train_t = dead_ID_use[train_t]
            id_val_test_t = dead_ID_use[test_t]
            id_train_f = not_dead_ID_use[train_f]
            id_val_test_f = not_dead_ID_use[test_f]
            id_val_t = id_val_test_t.sample(frac=0.5, random_state=RANDOM_SEED)
            id_test_t = id_val_test_t.drop(id_val_t.index)
            id_val_f = id_val_test_f.sample(frac=0.5, random_state=RANDOM_SEED)
            id_test_f = id_val_test_f.drop(id_val_f.index)

            id_test = pd.concat([id_test_t, id_test_f])
            test_id_label = pd.DataFrame(data=list(zip(id_test, [1] * len(id_test_t) + [0] * len(id_test_f))),
                                         columns=['id', 'label'])

            id_val = pd.concat([id_val_t, id_val_f])
            val_id_label = pd.DataFrame(data=list(zip(id_val, [1] * len(id_val_t) + [0] * len(id_val_f))),
                                        columns=['id', 'label'])

            id_train = pd.concat([id_train_t, id_train_f])
            train_id_label = pd.DataFrame(data=list(zip(id_train, [1] * len(id_train_t) + [0] * len(id_train_f))),
                                          columns=['id', 'label'])

            mortality_train = df[df.Adm_ID.isin(train_id_label.id)]
            mortality_val = df[df.Adm_ID.isin(val_id_label.id)]
            mortality_test = df[df.Adm_ID.isin(test_id_label.id)]
            mortality_not_use = df[
                (~df.Adm_ID.isin(train_id_label.id)) & (
                            ~df.Adm_ID.isin(val_id_label.id) & (~df.Adm_ID.isin(test_id_label.id)))]

            train_result = mortality_train.Label.value_counts()

            val_result = mortality_val.Label.value_counts()

            test_result = mortality_test.Label.value_counts()

            no_result = mortality_not_use.Label.value_counts()

            if len(no_result) == 2:
                write_log(("In the {}th split of {} folds\n"
                           "In the train dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                           "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                           "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                           "In the not use dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}").format(
                    num,
                    args.Kfold,
                    train_result[1],
                    train_result[0],
                    val_result[1],
                    val_result[0],
                    test_result[1],
                    test_result[0],
                    no_result[1],
                    no_result[0]),
                    LOG_PATH)
            else:
                write_log(("In the {}th split of {} folds\n"
                           "In the train dataset Positive Patients' Notes: {}, Negative  Patients' Notes: {}\n"
                           "In the validation dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                           "In the test dataset Positive Patients' Notes: {}, Negative Patients' Notes: {}\n"
                           "In the not use dataset Negative  Patients' Notes: {}").format(num,
                                                                                          args.Kfold,
                                                                                          train_result[1],
                                                                                          train_result[0],
                                                                                          val_result[1],
                                                                                          val_result[0],
                                                                                          test_result[1],
                                                                                          test_result[0],
                                                                                          no_result[0]),
                          LOG_PATH)

            os.makedirs(os.path.join(args.output_dir, str(num)))
            mortality_train.to_csv(os.path.join(args.output_dir, str(num), 'train.csv'), index=False)
            mortality_val.to_csv(os.path.join(args.output_dir, str(num), 'val.csv'), index=False)
            mortality_test.to_csv(os.path.join(args.output_dir, str(num), 'test.csv'), index=False)
            mortality_not_use.to_csv(os.path.join(args.output_dir, str(num), 'not_use.csv'), index=False)
            df.to_csv(os.path.join(args.output_dir, str(num), 'full.csv'), index=False)

            write_log("Data saved in the {}".format(args.output_dir), str(num), LOG_PATH)


if __name__ == "__main__":
    main()