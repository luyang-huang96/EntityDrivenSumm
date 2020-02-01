""" full training (train rnn-ext + abs + RL) """
from training import BasicTrainer
import argparse
import json
import pickle as pkl
import os
from os.path import join, exists
from itertools import cycle

from toolz.sandbox.core import unzip
from cytoolz import identity

import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data.data import CnnDmDataset
from data.batcher import tokenize

from model.rl import ActorCritic, SelfCritic, SelfCriticEntity
from model.extract import PtrExtractSumm, PtrExtractSummEntity


from rl import get_grad_fn
from rl import A2CPipeline, SCPipeline
from decoding import load_best_ckpt
from decoding import Abstractor, ArticleBatcher
from metric import compute_rouge_l, compute_rouge_n, compute_rouge_l_summ


MAX_ABS_LEN = 100

try:
    DATA_DIR = os.environ['DATA']
except KeyError:
    print('please use environment variable to specify data directories')


class RLDataset(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        return art_sents, abs_sents

class RLDataset_entity(CnnDmDataset):
    """ get the article sentences only (for decoding use)"""
    def __init__(self, split, key='filtered_rule1_input_mention_cluster'):
        super().__init__(split, DATA_DIR)
        self.key = key
        print('using key: ', key)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents = js_data['article']
        abs_sents = js_data['abstract']
        input_clusters = js_data[self.key]
        return art_sents, abs_sents, input_clusters

def load_ext_net(ext_dir):
    ext_meta = json.load(open(join(ext_dir, 'meta.json')))
    assert ext_meta['net'] == 'ml_rnn_extractor' or ext_meta['net'] == "ml_entity_extractor"
    ext_ckpt = load_best_ckpt(ext_dir)
    ext_args = ext_meta['net_args']
    vocab = pkl.load(open(join(ext_dir, 'vocab.pkl'), 'rb'))
    if ext_meta['net'] == 'ml_rnn_extractor':
        ext = PtrExtractSumm(**ext_args)
    elif ext_meta['net'] == "ml_entity_extractor":
        ext = PtrExtractSummEntity(**ext_args)
    else:
        raise Exception('not implemented')
    ext.load_state_dict(ext_ckpt)
    return ext, vocab


def configure_net(abs_dir, ext_dir, cuda, sc, tv):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(ext_dir)
    if sc:
        agent = SelfCritic(extractor,
                           ArticleBatcher(agent_vocab, cuda),
                           time_variant=tv
        )
    else:
        agent = ActorCritic(extractor._sent_enc,
                        extractor._art_enc,
                        extractor._extractor,
                        ArticleBatcher(agent_vocab, cuda))
    if cuda:
        agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, abstractor, net_args

def configure_net_entity(abs_dir, ext_dir, cuda, sc, tv):
    """ load pretrained sub-modules and build the actor-critic network"""
    # load pretrained abstractor model
    if abs_dir is not None:
        abstractor = Abstractor(abs_dir, MAX_ABS_LEN, cuda)
    else:
        abstractor = identity

    # load ML trained extractor net and buiild RL agent
    extractor, agent_vocab = load_ext_net(ext_dir)
    if sc:
        agent = SelfCriticEntity(extractor,
                           ArticleBatcher(agent_vocab, cuda),
                           time_variant=tv
        )
    else:
        raise Exception('actor critic entity model not implemented')
    if cuda:
        agent = agent.cuda()

    net_args = {}
    net_args['abstractor'] = (None if abs_dir is None
                              else json.load(open(join(abs_dir, 'meta.json'))))
    net_args['extractor'] = json.load(open(join(ext_dir, 'meta.json')))

    return agent, agent_vocab, abstractor, net_args

def configure_training(opt, lr, clip_grad, lr_decay, batch_size,
                       gamma, reward, stop_coeff, stop_reward):
    assert opt in ['adam']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay
    train_params['gamma']          = gamma
    train_params['reward']         = reward
    train_params['stop_coeff']     = stop_coeff
    train_params['stop_reward']    = stop_reward

    return train_params

def build_batchers(batch_size):
    def coll(batch):
        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts = d
            return source_sents and extracts
        art_batch, abs_batch = unzip(batch)
        art_batch, abs_batch = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch)))))
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        return art_sents, abs_sents
    loader = DataLoader(
        RLDataset('train'), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset('val'), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader


def build_batchers_entity(batch_size, key):
    def coll(batch):
        split_token = '<split>'
        pad = 0
        art_batch, abs_batch, all_clusters = unzip(batch)
        art_sents = []
        abs_sents = []

        def is_good_data(d):
            """ make sure data is not empty"""
            source_sents, extracts = d
            return source_sents and extracts

        art_batch, abs_batch = list(zip(*list(filter(is_good_data, zip(art_batch, abs_batch)))))
        art_sents = list(filter(bool, map(tokenize(None), art_batch)))
        abs_sents = list(filter(bool, map(tokenize(None), abs_batch)))
        inputs = []
        # merge cluster
        for art_sent, clusters in zip(art_sents, all_clusters):
            cluster_words = []
            cluster_wpos = []
            cluster_spos = []
            for cluster in clusters:
                scluster_word = []
                scluster_wpos = []
                scluster_spos = []

                for mention in cluster:
                    if len(mention['text'].strip().split(' ')) == len(
                            list(range(mention['position'][3] + 1, mention['position'][4] + 1))):
                        scluster_word += mention['text'].lower().strip().split(' ')
                        scluster_wpos += list(range(mention['position'][3] + 1, mention['position'][4] + 1))
                        scluster_spos += [mention['position'][0] + 1 for _ in
                                          range(len(mention['text'].strip().split(' ')))]
                        scluster_word.append(split_token)
                        scluster_wpos.append(pad)
                        scluster_spos.append(pad)
                    else:
                        sent_num = mention['position'][0]
                        word_start = mention['position'][3]
                        word_end = mention['position'][4]
                        # if word_end > 99:
                        #     word_end = 99
                        if sent_num > len(art_sent) - 1:
                            print('bad cluster')
                            continue
                        scluster_word += art_sent[sent_num][word_start:word_end]
                        scluster_wpos += list(range(word_start, word_end))
                        scluster_spos += [mention['position'][0] + 1 for _ in
                                          range(word_start + 1, word_end + 1)]
                        scluster_word.append(split_token)
                        scluster_wpos.append(pad)
                        scluster_spos.append(pad)
                if scluster_word != []:
                    scluster_word.pop()
                    scluster_wpos.pop()
                    scluster_spos.pop()
                    cluster_words.append(scluster_word)
                    cluster_wpos.append(scluster_wpos)
                    cluster_spos.append(scluster_spos)
                    if len(scluster_word) != len(scluster_wpos):
                        print(scluster_word)
                        print(scluster_wpos)
                        print('cluster:', cluster)
                    if len(scluster_word) != len(scluster_spos):
                        print(scluster_word)
                        print(scluster_spos)
                        print('cluster:', cluster)
                    assert len(scluster_word) == len(scluster_spos) and len(scluster_spos) == len(scluster_wpos)


            new_clusters = (cluster_words, cluster_wpos, cluster_spos)
            inputs.append((art_sent, new_clusters))
        assert len(inputs) == len(abs_sents)
        return inputs, abs_sents
    if key == 1:
        key = 'filtered_rule1_input_mention_cluster'
    elif key == 2:
        key = 'filtered_rule23_6_input_mention_cluster'
    loader = DataLoader(
        RLDataset_entity('train', key), batch_size=batch_size,
        shuffle=True, num_workers=4,
        collate_fn=coll
    )
    val_loader = DataLoader(
        RLDataset_entity('val', key), batch_size=batch_size,
        shuffle=False, num_workers=4,
        collate_fn=coll
    )
    return cycle(loader), val_loader



def train(args):
    if not exists(args.path):
        os.makedirs(args.path)

    # make net
    if args.entity:
        agent, agent_vocab, abstractor, net_args = configure_net_entity(
            args.abs_dir, args.ext_dir, args.cuda, args.sc, args.tv)
    else:
        agent, agent_vocab, abstractor, net_args = configure_net(
            args.abs_dir, args.ext_dir, args.cuda, args.sc, args.tv)

    # configure training setting
    assert args.stop > 0
    train_params = configure_training(
        'adam', args.lr, args.clip, args.decay, args.batch,
        args.gamma, args.reward, args.stop, 'rouge-1'
    )
    if args.entity:
        train_batcher, val_batcher = build_batchers_entity(args.batch, args.key)
    else:
        train_batcher, val_batcher = build_batchers(args.batch)
    # TODO different reward
    if args.reward == 'rouge-l':
        reward_fn = compute_rouge_l
    elif args.reward == 'rouge-1':
        reward_fn = compute_rouge_n(n=1)
    elif args.reward == 'rouge-2':
        reward_fn = compute_rouge_n(n=2)
    elif args.reward == 'rouge-l-s':
        reward_fn = compute_rouge_l_summ
    else:
        raise Exception('Not prepared reward')
    stop_reward_fn = compute_rouge_n(n=1)

    # save abstractor binary
    if args.abs_dir is not None:
        abs_ckpt = {}
        abs_ckpt['state_dict'] = load_best_ckpt(args.abs_dir, reverse=True)
        abs_vocab = pkl.load(open(join(args.abs_dir, 'vocab.pkl'), 'rb'))
        abs_dir = join(args.path, 'abstractor')
        os.makedirs(join(abs_dir, 'ckpt'))
        with open(join(abs_dir, 'meta.json'), 'w') as f:
            json.dump(net_args['abstractor'], f, indent=4)
        torch.save(abs_ckpt, join(abs_dir, 'ckpt/ckpt-0-0'))
        with open(join(abs_dir, 'vocab.pkl'), 'wb') as f:
            pkl.dump(abs_vocab, f)
        # save configuration
    meta = {}
    meta['net']           = 'rnn-ext_abs_rl'
    meta['net_args']      = net_args
    meta['train_params']  = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)
    with open(join(args.path, 'agent_vocab.pkl'), 'wb') as f:
        pkl.dump(agent_vocab, f)

    # prepare trainer
    grad_fn = get_grad_fn(agent, args.clip)
    optimizer = optim.Adam(agent.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'max', verbose=True,
                                  factor=args.decay, min_lr=1e-5,
                                  patience=args.lr_p)
    if args.sc:
        pipeline = SCPipeline(meta['net'], agent, abstractor,
                               train_batcher, val_batcher,
                               optimizer, grad_fn,
                               reward_fn, args.entity)
    else:
        pipeline = A2CPipeline(meta['net'], agent, abstractor,
                           train_batcher, val_batcher,
                           optimizer, grad_fn,
                           reward_fn, args.gamma,
                           stop_reward_fn, args.stop)

    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler,
                           val_mode='score')

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='program to demo a Seq2Seq model'
    )
    parser.add_argument('--path', required=True, help='root of the model')
    parser.add_argument('--sc', action='store_true', help='use self critical')
    parser.add_argument('--tv', action='store_true', help='time variant sc')
    parser.add_argument('--entity', action='store_true', help='entity model')


    # model options
    parser.add_argument('--abs_dir', action='store',
                        help='pretrained summarizer model root path')
    parser.add_argument('--ext_dir', action='store',
                        help='root of the extractor model')
    parser.add_argument('--ckpt', type=int, action='store', default=None,
                        help='ckeckpoint used decode')
    parser.add_argument('--key', type=int, default=2, help='use which cluster type')

    # training options
    parser.add_argument('--reward', action='store', default='rouge-1',
                        help='reward function for RL')
    parser.add_argument('--lr', type=float, action='store', default=1e-4,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=2,
                        help='patience for learning rate decay')
    parser.add_argument('--gamma', type=float, action='store', default=0.95,
                        help='discount factor of RL')
    parser.add_argument('--stop', type=float, action='store', default=1.0,
                        help='stop coefficient for rouge-1')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=1000,
        help='number of update steps for che    ckpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    torch.cuda.set_device(args.gpu_id)

    train(args)
