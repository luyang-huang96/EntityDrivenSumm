""" run decoding of X-ext (+ abs)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time

from cytoolz import identity

import torch
from torch.utils.data import DataLoader

from data.batcher import tokenize, preproc

from decoding import Abstractor, Extractor, DecodeDataset, DecodeDatasetEntity, ExtractorEntity, BeamAbstractor
from decoding import make_html_safe
from nltk import sent_tokenize
from torch import multiprocessing as mp
from cytoolz import identity, concat, curry
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op


MAX_ABS_NUM = 6  # need to set max sentences to extract for non-RL extractor


def decode(save_path, abs_dir, ext_dir, split, batch_size, max_len, cuda, min_len):
    start = time()
    # setup model
    if abs_dir is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    else:
        #abstractor = Abstractor(abs_dir, max_len, cuda)
        abstractor = BeamAbstractor(abs_dir, max_len, cuda, min_len, reverse=args.reverse)
    if ext_dir is None:
        # NOTE: if no exstractor is provided then
        #       it would be  the lead-N extractor
        extractor = lambda art_sents: list(range(len(art_sents)))[:MAX_ABS_NUM]
    else:
        if args.no_force_ext:
            extractor = Extractor(ext_dir, max_ext=MAX_ABS_NUM, cuda=cuda, force_ext=not args.no_force_ext)
        else:
            extractor = Extractor(ext_dir, max_ext=MAX_ABS_NUM, cuda=cuda)

    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    os.makedirs(save_path)
    # prepare save paths and logs
    dec_log = {}
    dec_log['abstractor'] = (None if abs_dir is None
                             else json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['extractor'] = (None if ext_dir is None
                            else json.load(open(join(ext_dir, 'meta.json'))))
    dec_log['rl'] = False
    dec_log['split'] = split
    if abs_dir is not None:
        dec_log['beam'] = 5  # greedy decoding only
        beam_size = 5
    else:
        dec_log['beam'] = 1
        beam_size = 1
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)
    print(dec_log['extractor'])
    if dec_log['extractor']['net_args']['stop'] == False and not args.no_force_ext:
        for i in range(MAX_ABS_NUM+1):
            os.makedirs(join(save_path, 'output_{}'.format(i)))
    else:
        os.makedirs(join(save_path, 'output'))

    # Decoding
    i = 0
    length = 0
    with torch.no_grad():
        for i_debug, raw_article_batch in enumerate(loader):
            tokenized_article_batch = map(tokenize(None), raw_article_batch)
            ext_arts = []
            ext_inds = []
            pre_abs = []
            beam_inds = []
            for raw_art_sents in tokenized_article_batch:
                ext = extractor(raw_art_sents)
                ext_art = list(map(lambda i: raw_art_sents[i], ext))
                pre_abs.append([word for sent in ext_art for word in sent])
                beam_inds += [(len(beam_inds), 1)]

            if beam_size > 1:
                all_beams = abstractor(pre_abs, beam_size, diverse=1.0)
                dec_outs = rerank_mp(all_beams, beam_inds)
            else:
                dec_outs = abstractor(pre_abs)

            for dec_out in dec_outs:
                dec_out = sent_tokenize(' '.join(dec_out))
                ext = [sent.split(' ') for sent in dec_out]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += ext
            if dec_log['extractor']['net_args']['stop'] == False and not args.no_force_ext:
                dec_outs = ext_arts
                assert i == batch_size*i_debug
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                    for k, dec_str in enumerate(decoded_sents):
                        if k > MAX_ABS_NUM - 2:
                            break
                        with open(join(save_path, 'output_{}/{}.dec'.format(k, i)),
                                  'w') as f:
                            f.write(make_html_safe(dec_str))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i / n_data * 100, timedelta(seconds=int(time() - start))
                    ), end='')
            else:
                dec_outs = ext_arts
                assert i == batch_size * i_debug
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j + n]]
                    with open(join(save_path, 'output/{}.dec'.format(i)),
                              'w') as f:
                        f.write(make_html_safe('\n'.join(decoded_sents)))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i/n_data*100, timedelta(seconds=int(time()-start))
                    ), end='')
                    length += len(decoded_sents)
        print('average summary length:', length / i)


def decode_entity(save_path, abs_dir, ext_dir, split, batch_size, max_len, cuda, min_len):
    start = time()
    # setup model
    if abs_dir is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        abstractor = identity
    else:
        #abstractor = Abstractor(abs_dir, max_len, cuda)
        abstractor = BeamAbstractor(abs_dir, max_len, cuda, min_len, reverse=args.reverse)
    if ext_dir is None:
        # NOTE: if no exstractor is provided then
        #       it would be  the lead-N extractor
        raise Exception('do not use entity command')
    else:
        extractor = ExtractorEntity(ext_dir, max_ext=MAX_ABS_NUM, cuda=cuda)

    # setup loader
    def coll_entity(batch):
        batch = list(filter(bool, batch))
        return batch
    if args.key == 1:
        key = 'filtered_rule1_input_mention_cluster'
    elif args.key == 2:
        key = 'filtered_rule23_6_input_mention_cluster'
    else:
        raise Exception
    dataset = DecodeDatasetEntity(split, key=key)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll_entity
    )

    # prepare save paths and logs
    # for i in range(MAX_ABS_NUM):
    #     os.makedirs(join(save_path, 'output_{}'.format(i)))
    dec_log = {}
    dec_log['abstractor'] = (None if abs_dir is None
                             else json.load(open(join(abs_dir, 'meta.json'))))
    dec_log['extractor'] = (None if ext_dir is None
                            else json.load(open(join(ext_dir, 'meta.json'))))
    dec_log['rl'] = False
    dec_log['split'] = split
    # dec_log['beam'] = 1  # greedy decoding only
    if abs_dir is not None:
        dec_log['beam'] = 5  # greedy decoding only
        beam_size = 5
    else:
        dec_log['beam'] = 1
        beam_size = 1
    print(dec_log['extractor'])
    if dec_log['extractor']['net_args']['stop'] == False and not args.no_force_ext:
        for i in range(MAX_ABS_NUM):
            os.makedirs(join(save_path, 'output_{}'.format(i)))
    else:
        os.makedirs(join(save_path, 'output'))
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    i = 0
    length = 0
    with torch.no_grad():
        for i_debug, batch_data in enumerate(loader):
            raw_article_batch, raw_clusters = zip(*batch_data)
            tokenized_article_batch = list(map(tokenize(None), raw_article_batch))
            #processed_clusters = list(map(preproc, raw_clusters))
            ext_arts = []
            ext_inds = []
            pre_abs = []
            beam_inds = []
            for raw_art_sents, raw_cls in zip(tokenized_article_batch, raw_clusters):
                processed_cls = preproc(raw_art_sents, raw_cls)
                ext = extractor(raw_art_sents, processed_cls)
                ext_art = list(map(lambda i: raw_art_sents[i], ext))
                pre_abs.append([word for sent in ext_art for word in sent])
                beam_inds += [(len(beam_inds), 1)]

            if beam_size > 1:
                all_beams = abstractor(pre_abs, beam_size, diverse=1.0)
                dec_outs = rerank_mp(all_beams, beam_inds)
            else:
                dec_outs = abstractor(pre_abs)

            for dec_out in dec_outs:
                dec_out = sent_tokenize(' '.join(dec_out))
                ext = [sent.split(' ') for sent in dec_out]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += ext

            if dec_log['extractor']['net_args']['stop'] == False and not args.no_force_ext:
                #dec_outs = abstractor(ext_arts)
                dec_outs = ext_arts
                assert i == batch_size*i_debug
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                    for k, dec_str in enumerate(decoded_sents):
                        with open(join(save_path, 'output_{}/{}.dec'.format(k, i)),
                                  'w') as f:
                            f.write(make_html_safe(dec_str))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i / n_data * 100, timedelta(seconds=int(time() - start))
                    ), end='')
            else:
                dec_outs = ext_arts
                assert i == batch_size * i_debug
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j + n]]
                    with open(join(save_path, 'output/{}.dec'.format(i)),
                              'w') as f:
                        f.write(make_html_safe('\n'.join(decoded_sents)))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i/n_data*100, timedelta(seconds=int(time()-start))
                    ), end='')
                    length += len(decoded_sents)
    print('average summary length:', length / i)

_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)


def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def length_wu(cur_len, alpha=0.):
    """GNMT length re-ranking score.
    See "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    return ((5 + cur_len) / 6.0) ** alpha

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _compute_score(hyps):
    # all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    # # repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    # lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    try:
        lp = sum(h.logprob for h in hyps) / sum(length_wu(len(h.sequence)+1, alpha=0.9) for h in hyps)
    except ZeroDivisionError:
        lp = -1e5
    return lp


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description=('combine an extractor and an abstractor '
                     'to decode summary and evaluate on the '
                     'CNN/Daily Mail dataset')
    )
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--abs_dir', help='root of the abstractor model')
    parser.add_argument('--ext_dir', help='root of the extractor model')
    parser.add_argument('--entity', action='store_true', help='if model contains entity encoder')
    parser.add_argument('--key', type=int, default=2, help='use which cluster type')
    parser.add_argument('--reverse', action='store_true', help='if true then abstractor is trained with rl')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--n_ext', type=int, action='store', default=4,
                        help='number of sents to be extracted')
    parser.add_argument('--max_dec_word', type=int, action='store', default=100,
                        help='maximun words to be decoded for the abstractor')
    parser.add_argument('--no_force_ext', action='store_true', help='force extract same number of sents')
    parser.add_argument('--min_dec_word', type=int, action='store', default=35,
                        help='maximun words to be decoded for the abstractor')


    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    print(args)
    if args.entity == True:
        decode_entity(args.path, args.abs_dir, args.ext_dir,
               data_split, args.batch, args.max_dec_word, args.cuda, args.min_dec_word)
    else:
        decode(args.path, args.abs_dir, args.ext_dir,
           data_split, args.batch, args.max_dec_word, args.cuda, args.min_dec_word)
