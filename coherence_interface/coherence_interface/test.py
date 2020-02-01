# File:
# -*- coding: utf-8 -*-
# @Time    : 4/15/2019 5:50 PM
# @Author  : Derek Hu

import time
start = time.perf_counter()
import tensorflow as tf
from pairmatch_model import SeqMatchNet
import json
from utils import *
import argparse
import pickle

def add_arguments(parser):
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
    parser.add_argument("--learning_rate", type=float, default=0.15, help="Learning rate.")
    parser.add_argument('--use_glove', default=False, type=bool, help='use pretrained wordvec')
    parser.add_argument('--sm_conv1d_filter', default=128, type=int, help='filter num for conv1d')
    parser.add_argument('--sm_conv1d_width', default=3, type=int, help='filter width for conv1d')
    parser.add_argument('--sm_conv_filters', default=256, type=int, help='filter num for conv2d')
    parser.add_argument('--sm_conv_heights', default=3, type=int, help='filter height for conv2d')
    parser.add_argument('--sm_conv_widths', default=3, type=int, help='filter width for conv2d')
    parser.add_argument('--sm_maxpool_width', default=3, type=int, help='width for maxpooling')
    parser.add_argument('--sm_fc_num_units', default=256, type=int, help='Number of units in FC layers.')
    parser.add_argument('--sm_margin', default=1.0, type=int, help='Margin for ranking loss')

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()

model_path = "./save_model/nyt_new_pairwise/"

print("---load test data---")
data_path = "/data2/zhe/coherence/data_for_coherence_model/entity_distance_data_type2/process/"
test_data = [json.loads(line) for line in 
           open(data_path + "val_processed.jsonlist").readlines()]

print("num of test set: {}".format(len(test_data)))

print("step2: build reverse dict")
word_dict = pickle.load(open(data_path + "vocab.pickle", 'rb'))
reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    print("Loading saved model...")
    model = SeqMatchNet(trained_wordvec=False, word_vector=None, vocab_size=len(word_dict), args=args)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(model_path)
    print("model path: ", ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    batches = batch_iter_json(test_data, batch_size=32, num_epochs=1, shuffle=False)
    num_val = len(test_data)
    acc_val = 0
    loss_val = 0
    output_result = []
    for val_batch_data, _ in batches:
        num_batch = len(val_batch_data)
        val_batch_sentA = [np.array(elem["SA_idx"]) for elem in val_batch_data]
        val_batch_sentBP = [np.array(elem["SBP_idx"]) for elem in val_batch_data]
        val_batch_sentBN = [np.array(elem["SBN_idx"]) for elem in val_batch_data]
        val_batch_len_sentA = list(map(lambda x: len([y for y in x if y != 0]), val_batch_sentA))
        val_batch_len_sentBP = list(map(lambda x: len([y for y in x if y != 0]), val_batch_sentBP))
        val_batch_len_sentBN = list(map(lambda x: len([y for y in x if y != 0]), val_batch_sentBN))
        assert len(val_batch_sentA) == len(val_batch_sentBP) == len(val_batch_sentBN) == \
               len(val_batch_len_sentA) == len(val_batch_len_sentBP) == len(val_batch_len_sentBN)

        val_feed_dict = {
            model._sents_A_pos: val_batch_sentA,
            model._lengths_A_pos: val_batch_len_sentA,
            model._sents_A_neg: val_batch_sentA,
            model._lengths_A_neg: val_batch_len_sentA,
            model._sents_B_pos: val_batch_sentBP,
            model._lengths_B_pos: val_batch_len_sentBP,
            model._sents_B_neg: val_batch_sentBN,
            model._lengths_B_neg: val_batch_len_sentBN,
            model.dropout_rate_prob: 0,
            model.is_train: False
        }

        to_return = [model._loss, model._accuracy, model._output_pos, model._output_neg]

        loss_, acc_, output_pos_, output_neg_ = sess.run(to_return,
                                feed_dict=val_feed_dict)
        acc_val += acc_ * num_batch
        loss_val += loss_ * num_batch
        for idx, elem in enumerate(val_batch_data):
            sent_A = " ".join([reversed_dict[y] for y in list(val_batch_sentA[idx]) if y != 0])
            sent_BP = " ".join([reversed_dict[y] for y in list(val_batch_sentBP[idx]) if y != 0])
            sent_BN = " ".join([reversed_dict[y] for y in list(val_batch_sentBN[idx]) if y != 0])

            cur_result = {"pos_score": float(output_pos_[idx]), "neg_score": float(output_neg_[idx]),
                          "sentA": sent_A, "sentBP": sent_BP, "sentBN": sent_BN}
            output_result.append(cur_result)
    print("validation loss: {}, validation accuracy: {}".format(loss_val / num_val, acc_val / num_val))
    
    # write the results
    with open("test_result_type2.txt", 'w') as f_write:
        for elem in output_result:
            f_write.write("<sent A> " + elem["sentA"]  + '\n')
            f_write.write("<sent BP> " + elem["sentBP"] + " <score> " + str(elem["pos_score"])  + '\n')
            f_write.write("<sent BN> " + elem["sentBN"] + " <score> " + str(elem["neg_score"])  + '\n')
            f_write.write("====" * 10 + "\n")
    
