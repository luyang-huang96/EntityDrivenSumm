""" train extractor (ML)"""
from training import get_basic_grad_fn, basic_validate
from training import BasicPipeline, BasicTrainer
import argparse
import json
import os
from os.path import join, exists
import pickle as pkl

from cytoolz import compose

from model.extract import ExtractSumm, PtrExtractSumm, PtrExtractSummEntity, NNSESumm
from model.util import sequence_loss, binary_sequence_loss


from utils import PAD, UNK
from utils import make_vocab, make_embedding, make_vocab_entity

from data.data import CnnDmDataset
from data.batcher import coll_fn_extract, prepro_fn_extract, prepro_fn_extract_entity, coll_fn_extract_entity, coll_fn_extract_hardattn
from data.batcher import convert_batch_extract_ff, batchify_fn_extract_ff, prepro_fn_extract_hardattn
from data.batcher import convert_batch_extract_ptr, batchify_fn_extract_ptr
from data.batcher import convert_batch_extract_ptr_stop, batchify_fn_extract_nnse, convert_batch_extract_ptr_entity_stop
from data.batcher import batchify_fn_extract_ptr_entity, convert_batch_extract_ptr_entity, batchify_fn_extract_ptr_hardattn, convert_batch_extract_ptr_entity_hardattn
from data.batcher import BucketedGenerater

import torch
from torch import optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

BUCKET_SIZE = 6400

try:
    DATA_DIR = os.environ['DATA']
    print('data_dir:', DATA_DIR)
except KeyError:
    print('please use environment variable to specify data directories')

class ExtractDataset(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted']
        return art_sents, extracts

class ExtractDataset_neusum(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted_neu']
        return art_sents, extracts

class ExtractDataset_combine(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts = js_data['article'], js_data['extracted_combine']
        extracts = sorted(extracts)
        return art_sents, extracts


class EntityExtractDataset(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts, clusters = js_data['article'], js_data['extracted'], js_data['filtered_rule1_input_mention_cluster']
        return art_sents, extracts, clusters

class EntityExtractDataset_neusum(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split):
        super().__init__(split, DATA_DIR)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts, clusters = js_data['article'], js_data['extracted_neu'], js_data['filtered_rule1_input_mention_cluster']
        return art_sents, extracts, clusters

class EntityExtractDataset_combine(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, key='filtered_rule23_6_input_mention_cluster'):
        super().__init__(split, DATA_DIR)
        self.key = key
        print('using key:', key)

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        art_sents, extracts, clusters = js_data['article'], js_data['extracted_combine'], js_data[self.key]
        extracts = sorted(extracts)
        return art_sents, extracts, clusters

class EntityExtractDataset_hardattn(CnnDmDataset):
    """ article sentences -> extraction indices
    (dataset created by greedily matching ROUGE)
    """
    def __init__(self, split, key='filtered_rule23_6_input_mention_cluster'):
        super().__init__(split, DATA_DIR)
        self.key = key
        print('using key:', key)
        if self.key == 'filtered_rule1_input_mention_cluster':
            self.label_key = 'entity_label_rule1'
        elif self.key == 'filtered_rule23_6_input_mention_cluster':
            self.label_key = 'entity_label_rule23'

    def __getitem__(self, i):
        js_data = super().__getitem__(i)
        try:
            art_sents, extracts, clusters, cluster_label = js_data['article'], js_data['extracted_combine'], \
                                                           js_data[self.key], js_data[
                                                               self.label_key]
            if extracts != [] and cluster_label != []:
                extracts, cluster_label = list(zip(*sorted(zip(extracts, cluster_label), key=lambda x: x[0])))
                cluster_label = list(cluster_label)
                extracts = list(extracts)
                assert len(extracts) == len(cluster_label)
        except KeyError:
            art_sents, extracts, clusters = js_data['article'], js_data['extracted_combine'], \
                                                           js_data['filtered_rule1_input_mention_cluster']
            cluster_label = []
        return art_sents, extracts, clusters, cluster_label


def build_batchers(net_type, word2id, cuda, debug):
    assert net_type in ['ff', 'rnn', 'nnse']
    prepro = prepro_fn_extract(args.max_word, args.max_sent)
    def sort_key(sample):
        src_sents, _ = sample
        return len(src_sents)

    batchify_fn = (batchify_fn_extract_ff if net_type == 'ff'
                   else batchify_fn_extract_ptr)
    if net_type == 'nnse':
        batchify_fn = batchify_fn_extract_nnse
    convert_batch = (convert_batch_extract_ff if net_type in ['ff', 'nnse']
                     else convert_batch_extract_ptr)

    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id))

    train_loader = DataLoader(
        ExtractDataset_combine('train'), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        ExtractDataset_combine('val'), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)

    return train_batcher, val_batcher

def build_batchers_entity(net_type, word2id, cuda, debug):
    assert net_type in ['entity']

    prepro = prepro_fn_extract_entity(args.max_word, args.max_sent)

    # def sort_key(sample):
    #     src_sents, _, _ = sample
    #     return len(src_sents)
    def sort_key(sample):
        src_sents = sample[0]
        return len(src_sents)


    key = 'filtered_rule23_6_input_mention_cluster'


    batchify_fn = batchify_fn_extract_ptr_entity
    convert_batch = convert_batch_extract_ptr_entity


    batchify = compose(batchify_fn(PAD, cuda=cuda),
                       convert_batch(UNK, word2id))

    train_loader = DataLoader(
        EntityExtractDataset_combine('train', key), batch_size=BUCKET_SIZE,
        shuffle=not debug,
        num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract_entity
    )
    train_batcher = BucketedGenerater(train_loader, prepro, sort_key, batchify,
                                      single_run=False, fork=not debug)

    val_loader = DataLoader(
        EntityExtractDataset_combine('val', key), batch_size=BUCKET_SIZE,
        shuffle=False, num_workers=4 if cuda and not debug else 0,
        collate_fn=coll_fn_extract_entity
    )
    val_batcher = BucketedGenerater(val_loader, prepro, sort_key, batchify,
                                    single_run=True, fork=not debug)

    return train_batcher, val_batcher


def configure_net(net_type, vocab_size, emb_dim, conv_hidden,
                  lstm_hidden, lstm_layer, bidirectional):
    assert net_type in ['ff', 'rnn', 'nnse']
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['conv_hidden']   = conv_hidden
    net_args['lstm_hidden']   = lstm_hidden
    net_args['lstm_layer']    = lstm_layer
    net_args['bidirectional'] = bidirectional


    if net_type in ['ff', 'rnn']:
        net = (ExtractSumm(**net_args) if net_type == 'ff'
           else PtrExtractSumm(**net_args))
    elif net_type == 'nnse':
        net = NNSESumm(**net_args)
    return net, net_args

def configure_net_entity(net_type, vocab_size, emb_dim, conv_hidden,
                  lstm_hidden, lstm_layer, bidirectional):
    assert net_type in ['entity']
    net_args = {}
    net_args['vocab_size']    = vocab_size
    net_args['emb_dim']       = emb_dim
    net_args['conv_hidden']   = conv_hidden
    net_args['lstm_hidden']   = lstm_hidden
    net_args['lstm_layer']    = lstm_layer
    net_args['bidirectional'] = bidirectional

    net = PtrExtractSummEntity(**net_args)
    print('net:', net)
    return net, net_args


def configure_training(net_type, opt, lr, clip_grad, lr_decay, batch_size, hard_attention):
    """ supports Adam optimizer only"""
    assert opt in ['adam']
    assert net_type in ['ff', 'rnn', 'entity', 'nnse']
    opt_kwargs = {}
    opt_kwargs['lr'] = lr

    train_params = {}
    train_params['optimizer']      = (opt, opt_kwargs)
    train_params['clip_grad_norm'] = clip_grad
    train_params['batch_size']     = batch_size
    train_params['lr_decay']       = lr_decay

    if not hard_attention:
        print('general loss')
        if net_type == 'ff':
            criterion = lambda logit, target: F.binary_cross_entropy_with_logits(
                logit, target, reduce=False)
        elif net_type == 'nnse':
            criterion = lambda logit, target: F.binary_cross_entropy_with_logits(
                logit, target, reduce=False)
        else:
            ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
            def criterion(logits, targets):
                return sequence_loss(logits, targets, ce, pad_idx=-1)
    else:
        print('Two loss!')
        ce = lambda logit, target: F.cross_entropy(logit, target, reduce=False)
        bce = lambda logit, target: F.binary_cross_entropy_with_logits(logit, target, reduce=False)
        def criterion(logit1_sent, logit_en, target_sent, target_en):
            sent_loss = sequence_loss(logit1_sent, target_sent, ce, pad_idx=-1)
            #entity_loss = F.binary_cross_entropy_with_logits(logit_en, target_en)
            print('logit_en:', logit_en)
            print('target_en:', target_en)
            entity_loss = binary_sequence_loss(logit_en, target_en, bce, pad_idx=-1)
            print('entity loss: {:.4f}'.format(entity_loss.mean().item()), end=' ')
            loss = sent_loss.mean() + entity_loss.mean()
            del entity_loss, sent_loss
            return loss

    return criterion, train_params


def main(args):
    assert args.net_type in ['ff', 'rnn', 'entity', 'nnse']
    # create data batcher, vocabulary
    # batcher
    with open(join(DATA_DIR, 'vocab_cnt.pkl'), 'rb') as f:
        wc = pkl.load(f)
    if args.net_type == 'entity':

        print('Using Entity Encoder')
        word2id = make_vocab_entity(wc, args.vsize)
        train_batcher, val_batcher = build_batchers_entity(args.net_type, word2id,
                                                    args.cuda, args.debug)
        # make net
        net, net_args = configure_net_entity(args.net_type,
                                             len(word2id), args.emb_dim, args.conv_hidden,
                                             args.lstm_hidden, args.lstm_layer, args.bi)
    else:
        word2id = make_vocab(wc, args.vsize)
        train_batcher, val_batcher = build_batchers(args.net_type, word2id,
                                                args.cuda, args.debug)
        net, net_args = configure_net(args.net_type,
                                      len(word2id), args.emb_dim, args.conv_hidden,
                                      args.lstm_hidden, args.lstm_layer, args.bi)


    if args.w2v:
        # NOTE: the pretrained embedding having the same dimension
        #       as args.emb_dim should already be trained
        embedding, oovs = make_embedding(
            {i: w for w, i in word2id.items()}, args.w2v)
        print('oovs:', oovs)
        net.set_embedding(embedding)

    # configure training setting
    criterion, train_params = configure_training(
        args.net_type, 'adam', args.lr, args.clip, args.decay, args.batch, args.hard_attention
    )

    # save experiment setting
    if not exists(args.path):
        os.makedirs(args.path)
    with open(join(args.path, 'vocab.pkl'), 'wb') as f:
        pkl.dump(word2id, f, pkl.HIGHEST_PROTOCOL)
    meta = {}
    meta['net']           = 'ml_{}_extractor'.format(args.net_type)
    meta['net_args']      = net_args
    meta['traing_params'] = train_params
    with open(join(args.path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=4)

    # prepare trainer
    val_fn = basic_validate(net, criterion)
    grad_fn = get_basic_grad_fn(net, args.clip)
    optimizer = optim.Adam(net.parameters(), **train_params['optimizer'][1])
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True,
                                  factor=args.decay, min_lr=0,
                                  patience=args.lr_p)

    if args.cuda:
        net = net.cuda()
    pipeline = BasicPipeline(meta['net'], net,
                             train_batcher, val_batcher, args.batch, val_fn,
                             criterion, optimizer, grad_fn)
    trainer = BasicTrainer(pipeline, args.path,
                           args.ckpt_freq, args.patience, scheduler)

    print('start training with the following hyper-parameters:')
    print(meta)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the feed-forward extractor (ff-ext, ML)'
    )
    parser.add_argument('--path', required=True, help='root of the model')

    # model options
    parser.add_argument('--net-type', action='store', default='rnn',
                        help='model type of the extractor (ff/rnn), entity, nnse')
    parser.add_argument('--vsize', type=int, action='store', default=50000,
                        help='vocabulary size')
    parser.add_argument('--emb_dim', type=int, action='store', default=128,
                        help='the dimension of word embedding')
    parser.add_argument('--w2v', action='store',
                        help='use pretrained word2vec embedding')
    parser.add_argument('--conv_hidden', type=int, action='store', default=100,
                        help='the number of hidden units of Conv')
    parser.add_argument('--lstm_hidden', type=int, action='store', default=256,
                        help='the number of hidden units of lSTM')
    parser.add_argument('--lstm_layer', type=int, action='store', default=1,
                        help='the number of layers of LSTM Encoder')
    parser.add_argument('--no-bi', action='store_true',
                        help='disable bidirectional LSTM encoder')

    # length limit
    parser.add_argument('--max_word', type=int, action='store', default=100,
                        help='maximun words in a single article sentence')
    parser.add_argument('--max_sent', type=int, action='store', default=60,
                        help='maximun sentences in an article article')
    # training options
    parser.add_argument('--lr', type=float, action='store', default=1e-3,
                        help='learning rate')
    parser.add_argument('--decay', type=float, action='store', default=0.5,
                        help='learning rate decay ratio')
    parser.add_argument('--lr_p', type=int, action='store', default=0,
                        help='patience for learning rate decay')
    parser.add_argument('--clip', type=float, action='store', default=2.0,
                        help='gradient clipping')
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='the training batch size')
    parser.add_argument(
        '--ckpt_freq', type=int, action='store', default=3000,
        help='number of update steps for checkpoint and validation'
    )
    parser.add_argument('--patience', type=int, action='store', default=5,
                        help='patience for early stopping')

    parser.add_argument('--debug', action='store_true',
                        help='run in debugging mode')
    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    args.bi = not args.no_bi
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    torch.cuda.set_device(args.gpu_id)

    main(args)
