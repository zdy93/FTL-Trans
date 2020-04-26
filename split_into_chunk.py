#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""

import pandas as pd
from tqdm import trange, tqdm
import argparse
import os
from other_func import write_log, split_into_chunks


def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--train_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input training data file name."
                             " Should be the .tsv file (or other data file) for the task.")

    parser.add_argument("--val_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input validation data file name."
                             " Should be the .tsv file (or other data file) for the task.")

    parser.add_argument("--test_data",
                        default=None,
                        type=str,
                        required=True,
                        help="The input test data file name."
                             " Should be the .tsv file (or other data file) for the task.")

    parser.add_argument("--log_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The log file path.")

    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")

    args = parser.parse_args()
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    LOG_PATH = args.log_path
    MAX_LEN = args.max_seq_length

    write_log(("New Split Job Start! \n"
               "data_dir: {}, train_data: {}, val_data: {}, test_data: {} \n"
               "log_path: {}, output_dir: {}, max_seq_length: {}").format(args.data_dir, args.train_data,
                                                                        args.val_data, args.test_data,
                                                                        args.log_path, args.output_dir,
                                                                        args.max_seq_length), LOG_PATH)

    train_file_path = os.path.join(args.data_dir, args.train_data)
    val_file_path = os.path.join(args.data_dir, args.val_data)
    test_file_path = os.path.join(args.data_dir, args.test_data)
    train_df = pd.read_csv(train_file_path)
    val_df = pd.read_csv(val_file_path)
    test_df = pd.read_csv(test_file_path)

    new_train_df = split_into_chunks(train_df, MAX_LEN)
    new_val_df = split_into_chunks(val_df, MAX_LEN)
    new_test_df = split_into_chunks(test_df, MAX_LEN)

    train_result = new_train_df.Label.value_counts()
    val_result = new_val_df.Label.value_counts()
    test_result = new_test_df.Label.value_counts()

    write_log(("In the train dataset Positive Patients' Chunks: {}, Negative Patients' Chunks: {}\n"
               "In the validation dataset Positive Patients' Chunks: {}, Negative Patients' Chunks: {}\n"
               "In the test dataset Positive Patients' Chunks: {}, Negative Patients' Chunks: {}").format(train_result[1],
                                                                                                  train_result[0],
                                                                                                  val_result[1],
                                                                                                  val_result[0],
                                                                                                  test_result[1],
                                                                                                  test_result[0]),
              LOG_PATH)

    new_train_df.to_csv(os.path.join(args.output_dir, args.train_data), index=False)
    new_val_df.to_csv(os.path.join(args.output_dir, args.val_data), index=False)
    new_test_df.to_csv(os.path.join(args.output_dir, args.test_data), index=False)

    write_log("Split finished", LOG_PATH)


if __name__ == "__main__":
    main()