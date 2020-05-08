#!/usr/bin/env python3
#-*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""

import time
import os
import torch
import random
import argparse
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_transformers import BertTokenizer, BertConfig
from pytorch_pretrained_bert.optimization import BertAdam
from modeling_readmission import BertModel
from modeling_patient import FTLSTMLayer
from tqdm import tqdm, trange
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
from dotmap import DotMap
from torch import nn
import re
from other_func import write_log, Tokenize_with_note_id_hour, concat_by_id_list_with_note_chunk_id_time
from other_func import convert_note_ids, flat_accuracy, write_performance, reorder_by_time


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

    parser.add_argument("--save_model",
                        default=False,
                        action='store_true',
                        help="Whether to save the model.")

    parser.add_argument("--bert_model",
                        default="bert-base-uncased",
                        type=str,
                        required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")

    parser.add_argument("--embed_mode",
                        default=None,
                        type=str,
                        required=True,
                        help="The embedding type selected in the list: all, note, chunk, no.")

    parser.add_argument("--task_name",
                        default="FTLSTM_hour_with_ClBERT_mortality",
                        type=str,
                        required=True,
                        help="The name of the task.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_chunk_num",
                        default=64,
                        type=int,
                        help="The maximum total input chunk numbers after WordPiece tokenization.")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=2e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--warmup_proportion",
                        default=0.0,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--num_train_epochs",
                        default=3,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.save_model:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    LOG_PATH = args.log_path
    MAX_LEN = args.max_seq_length

    config = DotMap()
    config.hidden_dropout_prob = 0.1
    config.layer_norm_eps = 1e-12
    config.initializer_range = 0.02
    config.max_note_position_embedding = 1000
    config.max_chunk_position_embedding = 1000
    config.embed_mode = args.embed_mode
    config.layer_norm_eps = 1e-12
    config.hidden_size = 768
    config.lstm_layers = 1

    config.task_name = args.task_name

    write_log(("New Job Start! \n"
               "Data directory: {}, Directory Code: {}, Save Model: {}\n"
               "Output_dir: {}, Task Name: {}, embed_mode: {}\n"
               "max_seq_length: {},  max_chunk_num: {}\n"
               "train_batch_size: {}, eval_batch_size: {}\n"
               "learning_rate: {}, warmup_proportion: {}\n"
               "num_train_epochs: {}, seed: {}, gradient_accumulation_steps: {}\n"
               "FTLSTM Model's lstm_layers: {}").format(args.data_dir,
                                                      args.data_dir.split('_')[-1],
                                                      args.save_model,
                                                      args.output_dir,
                                                      config.task_name,
                                                      config.embed_mode,
                                                      args.max_seq_length,
                                                      args.max_chunk_num,
                                                      args.train_batch_size,
                                                      args.eval_batch_size,
                                                      args.learning_rate,
                                                      args.warmup_proportion,
                                                      args.num_train_epochs,
                                                      args.seed,
                                                      args.gradient_accumulation_steps,
                                                      config.lstm_layers),
              LOG_PATH)

    content = "config setting: \n"
    for k, v in config.items():
        content += "{}: {} \n".format(k, v)
    write_log(content, LOG_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    write_log("Number of GPU is {}".format(n_gpu), LOG_PATH)
    for i in range(n_gpu):
        write_log(("Device Name: {},"
                   "Device Capability: {}").format(torch.cuda.get_device_name(i),
                                                   torch.cuda.get_device_capability(i)), LOG_PATH)

    train_file_path = os.path.join(args.data_dir, args.train_data)
    val_file_path = os.path.join(args.data_dir, args.val_data)
    test_file_path = os.path.join(args.data_dir, args.test_data)
    train_df = pd.read_csv(train_file_path)
    val_df = pd.read_csv(val_file_path)
    test_df = pd.read_csv(test_file_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    write_log("Tokenize Start!", LOG_PATH)
    train_df = reorder_by_time(train_df)
    val_df = reorder_by_time(val_df)
    test_df = reorder_by_time(test_df)
    train_labels, train_inputs, train_masks, train_note_ids, train_times = Tokenize_with_note_id_hour(train_df, MAX_LEN, tokenizer)
    validation_labels, validation_inputs, validation_masks, validation_note_ids, validation_times = Tokenize_with_note_id_hour(val_df, MAX_LEN, tokenizer)
    test_labels, test_inputs, test_masks, test_note_ids, test_times = Tokenize_with_note_id_hour(test_df, MAX_LEN, tokenizer)
    write_log("Tokenize Finished!", LOG_PATH)
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    test_inputs = torch.tensor(test_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    test_labels = torch.tensor(test_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)
    test_masks = torch.tensor(test_masks)
    train_times = torch.tensor(train_times)
    validation_times = torch.tensor(validation_times)
    test_times = torch.tensor(test_times)
    write_log(("train dataset size is %d,\n"
               "validation dataset size is %d,\n"
               "test dataset size is %d") % (len(train_inputs), len(validation_inputs), len(test_inputs)), LOG_PATH)

    (train_labels, train_inputs,
     train_masks, train_ids,
     train_note_ids, train_chunk_ids, train_times) = concat_by_id_list_with_note_chunk_id_time(train_df, train_labels,
                                                                                               train_inputs,
                                                                                               train_masks,
                                                                                               train_note_ids,
                                                                                               train_times, MAX_LEN)
    (validation_labels, validation_inputs,
     validation_masks, validation_ids,
     validation_note_ids, validation_chunk_ids,
     validation_times) = concat_by_id_list_with_note_chunk_id_time(val_df, validation_labels,
                                                                   validation_inputs, validation_masks,
                                                                   validation_note_ids, validation_times,
                                                                   MAX_LEN)
    (test_labels, test_inputs,
     test_masks, test_ids,
     test_note_ids, test_chunk_ids, test_times) = concat_by_id_list_with_note_chunk_id_time(test_df, test_labels,
                                                                                       test_inputs, test_masks,
                                                                                       test_note_ids, test_times,
                                                                                       MAX_LEN)

    model = BertModel.from_pretrained(args.bert_model).to(device)
    model.to(device)
    lstm_layer = FTLSTMLayer(config=config, num_labels=1)
    lstm_layer.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
        lstm_layer = torch.nn.DataParallel(lstm_layer)
    param_optimizer = list(model.named_parameters()) + list(lstm_layer.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    num_train_steps = int(
        len(train_labels) / args.gradient_accumulation_steps * args.num_train_epochs)

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)
    start = time.time()
    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = args.num_train_epochs
    write_log("Training start!", LOG_PATH)
    # trange is a tqdm wrapper around the normal python range
    with torch.autograd.set_detect_anomaly(False):
        for epoch in trange(epochs, desc="Epoch"):

            # Training

            # Set our model to training mode (as opposed to evaluation mode)
            model.train()
            lstm_layer.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            tr_ids_num = len(train_ids)
            tr_batch_loss = []
            for step in range(tr_ids_num):
                b_input_ids = train_inputs[step][-args.max_chunk_num:, :].to(device)
                b_input_mask = train_masks[step][-args.max_chunk_num:, :].to(device)
                b_note_ids = train_note_ids[step][-args.max_chunk_num:]
                b_new_note_ids = convert_note_ids(b_note_ids).to(device)
                b_chunk_ids = train_chunk_ids[step][-args.max_chunk_num:].unsqueeze(0).to(device)
                b_labels = train_labels[step].to(device)
                b_labels.resize_((1))
                _, whole_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                whole_input = whole_output.unsqueeze(0)
                b_new_note_ids = b_new_note_ids.unsqueeze(0)
                b_times = train_times[step][-args.max_chunk_num:].unsqueeze(0).to(device)
                loss, pred = lstm_layer(whole_input, b_times, b_new_note_ids, b_chunk_ids, b_labels)

                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                tr_batch_loss.append(loss.item())


                # Backward pass
                loss.backward()
                # Update parameters and take a step using the computed gradient
                if (step + 1) % args.train_batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    train_loss_set.append(np.mean(tr_batch_loss))
                    tr_batch_loss = []
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                del _, whole_output, whole_input, b_input_ids, b_input_mask, b_labels, b_times
                torch.cuda.empty_cache()

            write_log("Train loss: {}".format(tr_loss / nb_tr_steps), LOG_PATH)
            # Validation

            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()
            lstm_layer.eval()

            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            # Evaluate data for one epoch
            ev_ids_num = len(validation_ids)
            for step in range(ev_ids_num):
                with torch.no_grad():
                    b_input_ids = validation_inputs[step][-args.max_chunk_num:, :].to(device)
                    b_input_mask = validation_masks[step][-args.max_chunk_num:, :].to(device)
                    b_note_ids = validation_note_ids[step][-args.max_chunk_num:]
                    b_new_note_ids = convert_note_ids(b_note_ids).to(device)
                    b_chunk_ids = validation_chunk_ids[step][-args.max_chunk_num:].unsqueeze(0).to(device)
                    b_labels = validation_labels[step].to(device)
                    b_labels.resize_((1))
                    _, whole_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                    whole_input = whole_output.unsqueeze(0)
                    b_new_note_ids = b_new_note_ids.unsqueeze(0)
                    b_times = validation_times[step][-args.max_chunk_num:].unsqueeze(0).to(device)
                    pred = lstm_layer(whole_input, b_times, b_new_note_ids, b_chunk_ids).detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = flat_accuracy(pred, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            write_log("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps), LOG_PATH)
            output_checkpoints_path = os.path.join(args.output_dir,
                                                   "bert_fine_tuned_with_note_checkpoint_%d.pt" % epoch)

            if args.save_model:
                if n_gpu > 1:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'lstm_layer_state_dict': lstm_layer.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    },
                        output_checkpoints_path)
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'lstm_layer_state_dict': lstm_layer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    },
                        output_checkpoints_path)
    end = time.time()
    write_log("total training time is: {}s".format(end - start), LOG_PATH)

    fig1 = plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Patient Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    if args.save_model:
        output_fig_path = os.path.join(args.output_dir, "bert_fine_tuned_with_note_training_loss.png")
        plt.savefig(output_fig_path, dpi=fig1.dpi)
        output_model_state_dict_path = os.path.join(args.output_dir, "bert_fine_tuned_with_note_state_dict.pt")
        if n_gpu > 1:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'lstm_layer_state_dict': lstm_layer.module.state_dict(),
            },
                output_model_state_dict_path)
        else:
            torch.save({
                'model_state_dict': model.state_dict(),
                'lstm_layer_state_dict': lstm_layer.state_dict(),
            },
                output_model_state_dict_path)
        write_log("Model saved!", LOG_PATH)
    else:
        output_fig_path = os.path.join(args.output_dir,
                                       "bert_fine_tuned_with_note_training_loss_{}_{}.png".format(args.seed,
                                                                                                  args.data_dir.split(
                                                                                                      '_')[-1]))
        plt.savefig(output_fig_path, dpi=fig1.dpi)
        write_log("Model not saved as required", LOG_PATH)

    # Prediction on test set

    # Put model in evaluation mode
    model.eval()
    lstm_layer.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    te_ids_num = len(test_ids)
    for step in range(te_ids_num):
        b_input_ids = test_inputs[step][-args.max_chunk_num:, :].to(device)
        b_input_mask = test_masks[step][-args.max_chunk_num:, :].to(device)
        b_note_ids = test_note_ids[step][-args.max_chunk_num:]
        b_new_note_ids = convert_note_ids(b_note_ids).to(device)
        b_chunk_ids = test_chunk_ids[step][-args.max_chunk_num:].unsqueeze(0).to(device)
        b_labels = test_labels[step].to(device)
        b_labels.resize_((1))
        with torch.no_grad():
            _, whole_output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            whole_input = whole_output.unsqueeze(0)
            b_new_note_ids = b_new_note_ids.unsqueeze(0)
            b_times = test_times[step][-args.max_chunk_num:].unsqueeze(0).to(device)
            pred = lstm_layer(whole_input, b_times, b_new_note_ids, b_chunk_ids).detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()[0]
        predictions.append(pred)
        true_labels.append(label_ids)

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_logits = [item for sublist in predictions for item in sublist]
    flat_predictions = np.asarray([1 if i else 0 for i in (np.array(flat_logits) >= 0.5)])
    flat_true_labels = np.asarray(true_labels)

    output_df = pd.DataFrame({'pred_prob': flat_logits,
                              'pred_label': flat_predictions,
                              'label': flat_true_labels,
                              'Adm_ID': test_ids})

    if args.save_model:
        output_df.to_csv(os.path.join(args.output_dir, 'test_predictions.csv'), index=False)
    else:
        output_df.to_csv(os.path.join(args.output_dir,
                                      'test_predictions_{}_{}.csv'.format(args.seed,
                                                                          args.data_dir.split('_')[-1])), index=False)

    write_performance(flat_true_labels, flat_predictions, flat_logits, config, args)


if __name__ == "__main__":
    main()