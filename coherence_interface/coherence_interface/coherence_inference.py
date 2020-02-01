# File: code for coherence inference
# -*- coding: utf-8 -*-
# @Time    : 4/17/2019 3:36 PM
# @Author  : Derek Hu
import time
start = time.perf_counter()
import tensorflow as tf
from coherence_interface.pairmatch_model import SeqMatchNet
#from pairmatch_model import SeqMatchNet
import json
from utils import *
import argparse
import pickle
from nltk.tokenize import word_tokenize
import numpy as np
import os
import random

def add_arguments(parser):
    parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
    parser.add_argument("--learning_rate", type=float, default=0.15, help="Learning rate.")
    parser.add_argument('--use_glove', default=False, type=bool, help='use pretrained wordvec')


def sample_summary(summ):
    """construct negative summary"""
    if len(summ) == 2:
        summ_index = list(range(len(summ)))
        return {"summary_pos": summ_index, "summary_neg": summ_index[::-1]}
    else:
        summ_index = list(range(len(summ)))
        """
        summ_neg_index = list(range(len(summ)))
        summ_neg_index[0], summ_neg_index[-1] = summ_neg_index[-1], summ_neg_index[0]

        summ_middle_index = list(range(len(summ)))
        summ_middle_index[0], summ_middle_index[1] = summ_middle_index[1], summ_middle_index[0]
        """
        summ_neg_index = summ_index[::-1]
        summ_middle_index = list(range(len(summ)))
        while summ_middle_index == summ_index or summ_middle_index == summ_neg_index:
            random.shuffle(summ_middle_index)
        
        return {"summary_pos": summ_index, "summary_middle": summ_middle_index,
                "summary_neg": summ_neg_index}

def build_dataset(tokenized_text, word_dict, max_len=50):
    """tokenize, word to id, and padding"""
    tokenized_text = [word_dict.get(word, word_dict["<unk>"]) for word in tokenized_text]
    # if not existed, return word_dict_en["<unk>"]
    tokenized_text = tokenized_text[:max_len]
    tokenized_text = tokenized_text + (max_len - len(tokenized_text)) * [word_dict["<padding>"]]
    return tokenized_text


def compute_coherence(sess, model, sents_A, sents_B, lengths_A, lengths_B):
    result = sess.run(
        model._output_pos,
        feed_dict={
            model._sents_A_pos: sents_A,
            model._sents_B_pos: sents_B,
            model._lengths_A_pos: lengths_A,
            model._lengths_B_pos: lengths_B,
            model.dropout_rate_prob: 0,
            model.is_train: False
        })
    return result


def infer(sentA, sentB, word_dict, model, sess):
    sentA_processed = [np.array(build_dataset(sentA.lower().split(" "), word_dict=word_dict))]
    sentB_processed = [np.array(build_dataset(sentB.lower().split(" "), word_dict=word_dict))]
    len_sentA = list(map(lambda x: len([y for y in x if y != 0]), sentA_processed))
    len_sentB = list(map(lambda x: len([y for y in x if y != 0]), sentB_processed))
    result = compute_coherence(sess, model, sentA_processed, sentB_processed, len_sentA, len_sentB)
    return result


def global_infer(docs, word_dict, model, sess):
    if len(docs) < 2:
        return "num of sents in doc: ".format(len(docs))
    score_list = []
    for i in range(len(docs) - 1):
        j = i + 1
        sent_i = docs[i]
        sent_j = docs[j]
        score = infer(sent_i, sent_j, word_dict, model, sess)
        score_list.append(float(score))
    return score_list


# def graph_init(gpu_mem_use=0.1, gpu_use=0):
#     # load model
#     parser = argparse.ArgumentParser()
#     add_arguments(parser)
#     args = parser.parse_args()
#     model_path = "./coherence_interface/save_model/nyt_new_pairwise/"
#     word_dict = pickle.load(open("./coherence_interface/save_model/nyt_new_pairwise/vocab.pickle", 'rb'))
#
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_use,)
#     config=tf.ConfigProto(gpu_options=gpu_options, device_count = {'GPU': gpu_use})
#     config.gpu_options.visible_device_list= str(gpu_use)
#     sess = tf.Session(config=config)
#     model = SeqMatchNet(trained_wordvec=False, word_vector=None, vocab_size=len(word_dict), args=args)
#     saver = tf.train.Saver(tf.global_variables())
#     ckpt = tf.train.get_checkpoint_state(model_path)
#     saver.restore(sess, ckpt.model_checkpoint_path)
#     return word_dict, model, sess

def graph_init(gpu_mem_use=0.2, gpu_use=0, model="cnndm"):
    print('coherence model:', model)
    def add_arguments(parser):
        parser.add_argument("--embedding_size", type=int, default=300, help="Word embedding size.")
        parser.add_argument("--learning_rate", type=float, default=0.15, help="Learning rate.")
        parser.add_argument('--use_glove', default=False, type=bool, help='use pretrained wordvec')
    # load model
    # parsers = argparse.ArgumentParser()
    # add_arguments(parsers)
    # argss = parsers.parse_args()
    if model == "nyt":
        model_path = "/home/zhe/workspace/coherence_interface/save_model/nyt_new_pairwise/"
        word_dict = pickle.load(open("/home/zhe/workspace/coherence_interface/save_model/nyt_new_pairwise/vocab.pickle", 'rb'))
    elif model == "cnndm":
        model_path = "/home/zhe/workspace/coherence_interface/save_model/cnndm_new_pairwise/"
        word_dict = pickle.load(open("/home/zhe/workspace/coherence_interface/save_model/cnndm_new_pairwise/vocab.pickle", 'rb'))
    else:
        raise ValueError("model must be either <nyt> or <cnndm>")

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_mem_use,)
    config=tf.ConfigProto(gpu_options=gpu_options, device_count = {'GPU': gpu_use})
    config.gpu_options.visible_device_list= str(gpu_use)
    sess = tf.Session(config=config)
    model = SeqMatchNet(trained_wordvec=False, word_vector=None, vocab_size=len(word_dict), args=None)
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    return word_dict, model, sess

def coherence_infer(docs, model, sess, word_dict, aggregate="mean"):
    """input: docs: list of strings, and each sentence (str) is tokenized by space"""
    # inference
    score_list = global_infer(docs, word_dict, model, sess)
    if aggregate == "mean":
        final_score = np.mean(score_list)
    if aggregate == "median":
        final_score = np.median(score_list)
    if aggregate == "max":
        final_score = np.max(score_list)
    else:
        final_score = np.mean(score_list)
    return final_score



###### test ######
if __name__ == '__main__':

    word_dict, model, sess = graph_init()

    input_summ = ["News analysis : Bush administration is saying publicly that it is ` pleased ' that North Korea has agreed to resume talks on nuclear disarmament .", "behind closed doors at White House ane State Dept , some say country 's nuclear test should be answered with isolation .", "Secretary of State Condoleezza Rice is coming under increased fire inside and outside administration from officials and experts who are skeptical about what diplomacy can achieve in this case"]

    score = coherence_infer(input_summ, model, sess, word_dict)
    print(score)



