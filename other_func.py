#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""
import os
import time
import re
import io
import torch
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
from sklearn.metrics import roc_curve, precision_recall_curve, \
    auc, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from utils import pad_sequences


def write_log(content, log_path, print_content=True):
    if os.path.exists(log_path):
        with open(log_path, 'a') as f:
            f.write("Time: " + time.ctime() + "\n")
            f.write(content + "\n")
            f.write("=====================\n")
    else:
        with open(log_path, 'w') as f:
            f.write("Time: " + time.ctime() + "\n")
            f.write(content + "\n")
            f.write("=====================\n")
    if print_content:
        print(content)


def preprocess1(x):
    y = re.sub('\\[(.*?)\\]', '', x)  # remove de-identified brackets
    y = re.sub('[0-9]+\.', '', y)  # remove 1.2. since the segmenter segments based on this
    y = re.sub('dr\.', 'doctor', y)
    y = re.sub('m\.d\.', 'md', y)
    y = re.sub('admission date:', '', y)
    y = re.sub('discharge date:', '', y)
    y = re.sub('--|__|==', '', y)
    return y


def preprocessing(df_less_n, tokenizer):
    df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()

    df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))

    sen = df_less_n['TEXT'].values
    tokenized_texts = [tokenizer.tokenize(x) for x in sen]
    print("First sentence tokenized")
    print(tokenized_texts[0])
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    df_less_n['Input_ID'] = input_ids
    return df_less_n[['Adm_ID', 'Note_ID', 'TEXT', 'Input_ID', 'Label', 'chartdate', 'charttime']]


def word_count_pre(df_less_n):
    df_less_n['TEXT'] = df_less_n['TEXT'].fillna(' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\n', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].str.replace('\r', ' ')
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(str.strip)
    df_less_n['TEXT'] = df_less_n['TEXT'].str.lower()
    df_less_n['TEXT'] = df_less_n['TEXT'].apply(lambda x: preprocess1(x))
    return df_less_n


def split_into_chunks(df, max_len):
    input_ids = df.Input_ID.apply(lambda x: x[1:-1].replace(' ', '').split(','))
    df_len = len(df)
    Adm_ID, Note_ID, Input_ID, Label, chartdate, charttime = [], [], [], [], [], []
    for i in tqdm(range(df_len)):
        x = input_ids[i]
        n = int(len(x) / (max_len - 2))
        for j in range(n):
            Adm_ID.append(df.Adm_ID[i])
            Note_ID.append(df.Note_ID[i])
            sub_ids = x[j * (max_len - 2): (j + 1) * (max_len - 2)]
            sub_ids.insert(0, '101')
            sub_ids.append('102')
            Input_ID.append(' '.join(sub_ids))
            Label.append(df.Label[i])
            chartdate.append(df.chartdate[i])
            charttime.append(df.charttime[i])
        if len(x) % (max_len - 2) > 10:
            Adm_ID.append(df.Adm_ID[i])
            Note_ID.append(df.Note_ID[i])
            sub_ids = x[-((len(x)) % (max_len - 2)):]
            sub_ids.insert(0, '101')
            sub_ids.append('102')
            Input_ID.append(' '.join(sub_ids))
            Label.append(df.Label[i])
            chartdate.append(df.chartdate[i])
            charttime.append(df.charttime[i])
    new_df = pd.DataFrame({'Adm_ID': Adm_ID,
                           'Note_ID': Note_ID,
                           'Input_ID': Input_ID,
                           'Label': Label,
                           'chartdate': chartdate,
                           'charttime': charttime})
    new_df = new_df.astype({'Adm_ID': 'int64', 'Note_ID': 'int64', 'Label': 'int64'})
    return new_df


def Tokenize(df, max_length, tokenizer):
    labels = df.Label.values
    if 'TEXT' in df.columns:
        sen = df.TEXT.values
        labels = df.Label.values
        sen = ["[CLS] " + x + " [SEP]" for x in sen]
        tokenized_texts = [tokenizer.tokenize(x) for x in sen]
        print("First sentence tokenized")
        print(tokenized_texts[0])
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    else:
        assert 'Input_ID' in df.columns
        input_ids = df.Input_ID.apply(lambda x: x.split(' '))
        input_ids = input_ids.apply(lambda x: [int(i) for i in x])
        input_ids = input_ids.values
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return labels, input_ids, attention_masks


def Tokenize_with_note_id(df, max_length, tokenizer):
    labels = df.Label.values
    note_ids = df.Note_ID.values
    if 'TEXT' in df.columns:
        sen = df.TEXT.values
        labels = df.Label.values
        sen = ["[CLS] " + x + " [SEP]" for x in sen]
        tokenized_texts = [tokenizer.tokenize(x) for x in sen]
        print("First sentence tokenized")
        print(tokenized_texts[0])
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    else:
        assert 'Input_ID' in df.columns
        input_ids = df.Input_ID.apply(lambda x: x.split(' '))
        input_ids = input_ids.apply(lambda x: [int(i) for i in x])
        input_ids = input_ids.values
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return labels, input_ids, attention_masks, note_ids


def Tokenize_with_note_id_time(df, max_length, tokenizer):
    labels = df.Label.values
    note_ids = df.Note_ID.values
    times = pd.to_datetime(df.chartdate.values)
    times = times - times.min()
    times = times.days.values
    if 'TEXT' in df.columns:
        sen = df.TEXT.values
        labels = df.Label.values
        sen = ["[CLS] " + x + " [SEP]" for x in sen]
        tokenized_texts = [tokenizer.tokenize(x) for x in sen]
        print("First sentence tokenized")
        print(tokenized_texts[0])
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    else:
        assert 'Input_ID' in df.columns
        input_ids = df.Input_ID.apply(lambda x: x.split(' '))
        input_ids = input_ids.apply(lambda x: [int(i) for i in x])
        input_ids = input_ids.values
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return labels, input_ids, attention_masks, note_ids, times


def Tokenize_with_note_id_hour(df, max_length, tokenizer):
    labels = df.Label.values
    note_ids = df.Note_ID.values
    times = pd.to_datetime(df.charttime.values)
    times = times - times.min()
    times = times / pd.Timedelta(days=1)
    if 'TEXT' in df.columns:
        sen = df.TEXT.values
        labels = df.Label.values
        sen = ["[CLS] " + x + " [SEP]" for x in sen]
        tokenized_texts = [tokenizer.tokenize(x) for x in sen]
        print("First sentence tokenized")
        print(tokenized_texts[0])
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    else:
        assert 'Input_ID' in df.columns
        input_ids = df.Input_ID.apply(lambda x: x.split(' '))
        input_ids = input_ids.apply(lambda x: [int(i) for i in x])
        input_ids = input_ids.values
    input_ids = pad_sequences(input_ids, maxlen=max_length, dtype="long", truncating="post", padding="post")
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    return labels, input_ids, attention_masks, note_ids, times


def reorder_by_time(data):
    data.chartdate = pd.to_datetime(data.chartdate)
    data.charttime = pd.to_datetime(data.charttime)
    data.loc[data.charttime.isna(), 'charttime'] = data[data.charttime.isna()].chartdate + pd.Timedelta(hours=23,
                                                                                                        minutes=59,
                                                                                                        seconds=59)
    data = data.sort_values(by=['Adm_ID', 'charttime', 'Note_ID'])
    data.reset_index(inplace=True)
    return data


def concat_by_id_list(df, labels, inputs, masks, str_len):
    final_labels, final_inputs, final_masks = [], [], []
    id_lists = df.Adm_ID.unique()
    for id in id_lists:
        id_ix = df.index[df.Adm_ID == id].to_list()
        final_inputs.append(inputs[id_ix])
        final_masks.append(masks[id_ix])
        final_labels.append(labels[id_ix].max())
    return final_labels, final_inputs, final_masks, id_lists


def concat_by_id_list_with_note_chunk_id(df, labels, inputs, masks, note_ids, str_len):
    final_labels, final_inputs, final_masks, final_note_ids, final_chunk_ids = [], [], [], [], []
    id_lists = df.Adm_ID.unique()
    for id in id_lists:
        id_ix = df.index[df.Adm_ID == id].to_list()
        final_inputs.append(inputs[id_ix])
        final_masks.append(masks[id_ix])
        final_labels.append(labels[id_ix].max())
        final_note_ids.append(note_ids[id_ix])
        final_chunk_ids.append(torch.tensor(list(range(len(id_ix)))[::-1]))
    return final_labels, final_inputs, final_masks, id_lists, final_note_ids, final_chunk_ids


def concat_by_id_list_with_note_chunk_id_time(df, labels, inputs, masks, note_ids, times, str_len):
    final_labels, final_inputs, final_masks, final_note_ids, final_chunk_ids, final_times = [], [], [], [], [], []
    id_lists = df.Adm_ID.unique()
    for id in id_lists:
        id_ix = df.index[df.Adm_ID == id].to_list()
        final_inputs.append(inputs[id_ix])
        final_masks.append(masks[id_ix])
        final_labels.append(labels[id_ix].max())
        final_note_ids.append(note_ids[id_ix])
        final_chunk_ids.append(torch.tensor(list(range(len(id_ix)))[::-1]))
        final_times.append(torch.tensor(np.concatenate([np.zeros(1), np.diff(times[id_ix])])))
    return final_labels, final_inputs, final_masks, id_lists, final_note_ids, final_chunk_ids, final_times


def convert_note_ids(note_ids):
    new_dict = dict(zip(pd.Series(note_ids).unique(), range(len(pd.Series(note_ids).unique()))[::-1]))
    new_ids = [new_dict[i] for i in note_ids]
    return torch.tensor(new_ids)


def flat_accuracy(preds, labels):
    pred_flat = np.asarray([1 if i else 0 for i in (preds.flatten() >= 0.5)])
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def model_auc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr, thresholds


def model_aupr(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    aupr_score = auc(recall, precision)
    return aupr_score, precision, recall, thresholds


def write_performance(flat_true_labels, flat_predictions, flat_logits, config, args):
    test_accuracy = accuracy_score(flat_true_labels, flat_predictions)

    test_f1 = f1_score(flat_true_labels, flat_predictions, average='binary')

    test_prec = precision_score(flat_true_labels, flat_predictions, average='binary')

    test_rec = recall_score(flat_true_labels, flat_predictions, average='binary')

    test_auc, _, _, _ = model_auc(flat_true_labels, flat_logits)

    test_mc = matthews_corrcoef(flat_true_labels, flat_predictions)

    test_aupr, _, _, _ = model_aupr(flat_true_labels, flat_logits)

    test_msl = args.max_seq_length

    test_seed = args.seed

    test_dir_code = args.data_dir.split('_')[-1]

    test_time = time.ctime()

    exp_path = "{}_{}_{}.csv".format(config.task_name, config.embed_mode, test_msl)

    header = "Len,Dir,Seed,Accuracy,F1_Score,Precision,Recall,AUC,MCC,AUPR,Time"
    content = "{},{},{},{},{},{},{},{},{},{},{}".format(test_msl,
                                                        test_dir_code,
                                                        test_seed,
                                                        test_accuracy,
                                                        test_f1,
                                                        test_prec,
                                                        test_rec,
                                                        test_auc,
                                                        test_mc,
                                                        test_aupr,
                                                        test_time)

    if os.path.exists(exp_path):
        with open(exp_path, 'a') as f:
            f.write(content + "\n")
    else:
        with open(exp_path, 'w') as f:
            f.write(header + "\n")
            f.write(content + "\n")

    write_log("Test Patient Level Accuracy: {}\n"
              "Test Patient Level F1 Score: {}\n"
              "Test Patient Level Precision: {}\n"
              "Test Patient Level Recall: {}\n"
              "Test Patient Level AUC: {} \n"
              "Test Patient Level Matthew's correlation coefficient: {}\n"
              "Test Patient Level AUPR: {} \n"
              "All Finished!".format(test_accuracy,
                                     test_f1,
                                     test_prec,
                                     test_rec,
                                     test_auc,
                                     test_mc,
                                     test_aupr), args.log_path)
