""" batching """
import random
from collections import defaultdict

from toolz.sandbox import unzip
from cytoolz import curry, concat, compose
from cytoolz import curried

import torch
import torch.multiprocessing as mp
from utils import PAD, UNK, START, END


# Batching functions
def coll_fn(data):
    source_lists, target_lists = unzip(data)
    # NOTE: independent filtering works because
    #       source and targets are matched properly by the Dataset
    sources = list(filter(bool, concat(source_lists)))
    targets = list(filter(bool, concat(target_lists)))
    assert all(sources) and all(targets)
    return sources, targets

def coll_fn_extract(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts = d
        return source_sents and extracts
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

def coll_fn_extract_entity(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts, clusters = d
        return (source_sents and extracts) and clusters
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

def coll_fn_extract_hardattn(data):
    def is_good_data(d):
        """ make sure data is not empty"""
        source_sents, extracts, clusters, labels = d
        return (source_sents and extracts) and (clusters and labels)
    batch = list(filter(is_good_data, data))
    assert all(map(is_good_data, batch))
    return batch

@curry
def tokenize(max_len, texts):
    return [t.strip().lower().split()[:max_len] for t in texts]

def conver2id(unk, word2id, words_list):
    word2id = defaultdict(lambda: unk, word2id)
    return [[word2id[w] for w in words] for words in words_list]

@curry
def prepro_fn(max_src_len, max_tgt_len, batch):
    sources, targets = batch
    sources = tokenize(max_src_len, sources)
    targets = tokenize(max_tgt_len, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def prepro_fn_extract(max_src_len, max_src_num, batch):
    def prepro_one(sample):
        source_sents, extracts = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        return tokenized_sents, cleaned_extracts
    batch = list(map(prepro_one, batch))
    return batch

@curry
def prepro_fn_extract_entity(max_src_len, max_src_num, batch, pad=0, split_token='<split>'):
    # split will be "split token"
    def prepro_one(sample):
        source_sents, extracts, clusters = sample
        tokenized_sents = tokenize(max_src_len, source_sents)[:max_src_num]
        cleaned_extracts = list(filter(lambda e: e < len(tokenized_sents),
                                       extracts))
        # merge cluster
        cluster_words = []
        cluster_wpos = []
        cluster_spos = []
        for cluster in clusters:
            scluster_word = []
            scluster_wpos = []
            scluster_spos = []
            for mention in cluster:
                if mention['position'][0] > max_src_num-2:
                    continue
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
                    if word_end > 99:
                        word_end = 99
                    scluster_word += tokenized_sents[sent_num][word_start:word_end]
                    scluster_wpos += list(range(word_start, word_end))
                    scluster_spos += [mention['position'][0] + 1 for _ in
                                      range(word_start+1, word_end+1)]
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
                assert len(scluster_word) == len(scluster_spos) and len(scluster_spos) == len(scluster_wpos)

        clusters = (cluster_words, cluster_wpos, cluster_spos)

        return tokenized_sents, cleaned_extracts, clusters
    batch = list(map(prepro_one, batch))
    return batch


@curry
def preproc(tokenized_sents, clusters):
    pad = 0
    split_token = '<split>'
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
                scluster_word += tokenized_sents[sent_num][word_start:word_end]
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
                continue
                # print(scluster_word)
                # print(scluster_wpos)
                # print('cluster:', cluster)
            assert len(scluster_word) == len(scluster_spos) and len(scluster_spos) == len(scluster_wpos)
    clusters = (cluster_words, cluster_wpos, cluster_spos)

    return clusters


@curry
def convert_batch(unk, word2id, batch):
    sources, targets = unzip(batch)
    sources = conver2id(unk, word2id, sources)
    targets = conver2id(unk, word2id, targets)
    batch = list(zip(sources, targets))
    return batch

@curry
def convert_batch_copy(unk, word2id, batch):
    sources, targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, sources)
    sources = conver2id(unk, word2id, sources)
    tar_ins = conver2id(unk, word2id, targets)
    targets = conver2id(unk, ext_word2id, targets)
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return batch


@curry
def convert_batch_extract_ptr(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_stop(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        extracts.append(len(source_sents))
        return id_sents, extracts
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ff(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts = sample
        id_sents = conver2id(unk, word2id, source_sents)
        binary_extracts = [0] * len(source_sents)
        for ext in extracts:
            binary_extracts[ext] = 1
        return id_sents, binary_extracts
    batch = list(map(convert_one, batch))
    return batch


@curry
def pad_batch_tensorize(inputs, pad, cuda=True, max_num=0):
    """pad_batch_tensorize

    :param inputs: List of size B containing torch tensors of shape [T, ...]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    batch_size = len(inputs)
    try:
        max_len = max(len(ids) for ids in inputs)
    except ValueError:
        # print('inputs:', inputs)
        # print('batch_size:', batch_size)
        if inputs == []:
            max_len = 1
            batch_size = 1
    if max_len < max_num:
        max_len = max_num
    tensor_shape = (batch_size, max_len)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    for i, ids in enumerate(inputs):
        tensor[i, :len(ids)] = tensor_type(ids)
    return tensor

@curry
def double_pad_batch_tensorize(inputs, pad, cuda=True):
    """pad_batch_tensorize double pad and turn it to one hot vector

    :param inputs: List of List of size B containing torch tensors of shape [[T, ...],]
    :type inputs: List[np.ndarray]
    :rtype: TorchTensor of size (B, T, ...)
    """
    #tensor_type = torch.cuda.LongTensor if cuda else torch.LongTensor
    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    batch_size = len(inputs)
    max_sent_num = max([len(labels) for labels in inputs])
    max_side_num = max([labels[-1][0] for labels in inputs]) + 1
    tensor_shape = (batch_size, max_sent_num, max_side_num)
    tensor = tensor_type(*tensor_shape)
    tensor.fill_(pad)
    if pad < 0:
        for batch_id, labels in enumerate(inputs):
            for sent_id, label in enumerate(labels):
                tensor[batch_id, sent_id, :] = 0
                for label_id in label:
                    tensor[batch_id, sent_id, label_id] = 1
    else:
        for batch_id, labels in enumerate(inputs):
            for sent_id, label in enumerate(labels):
                for label_id in label:
                    tensor[batch_id, sent_id, label_id] = 1
    return tensor

@curry
def batchify_fn(pad, start, end, data, cuda=True):
    sources, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    tar_ins = [[start] + tgt for tgt in targets]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)

    fw_args = (source, src_lens, tar_in)
    loss_args = (target, )
    return fw_args, loss_args


@curry
def batchify_fn_copy(pad, start, end, data, cuda=True):
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(data)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    fw_args = (source, src_lens, tar_in, ext_src, ext_vsize)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def convert_batch_copy_rl(unk, word2id, batch):
    raw_sources, raw_targets = map(list, unzip(batch))
    ext_word2id = dict(word2id)
    for source in raw_sources:
        for word in source:
            if word not in ext_word2id:
                ext_word2id[word] = len(ext_word2id)
    src_exts = conver2id(unk, ext_word2id, raw_sources)
    sources = conver2id(unk, word2id, raw_sources)
    tar_ins = conver2id(unk, word2id, raw_targets)
    targets = conver2id(unk, ext_word2id, raw_targets)
    batch = list(zip(sources, src_exts, tar_ins, targets))
    return (batch, ext_word2id, raw_sources, raw_targets)

@curry
def batchify_fn_copy_rl(pad, start, end, data, cuda=True):
    batch, ext_word2id, raw_articles, raw_targets = data
    sources, ext_srcs, tar_ins, targets = tuple(map(list, unzip(batch)))

    src_lens = [len(src) for src in sources]
    sources = [src for src in sources]
    ext_srcs = [ext for ext in ext_srcs]

    tar_ins = [[start] + tgt for tgt in tar_ins]
    targets = [tgt + [end] for tgt in targets]

    source = pad_batch_tensorize(sources, pad, cuda)
    tar_in = pad_batch_tensorize(tar_ins, pad, cuda)
    target = pad_batch_tensorize(targets, pad, cuda)
    ext_src = pad_batch_tensorize(ext_srcs, pad, cuda)

    ext_vsize = ext_src.max().item() + 1
    extend_vsize = len(ext_word2id)
    ext_id2word = {_id:_word for _word, _id in ext_word2id.items()}
    #print('ext_size:', ext_vsize, extend_vsize)
    fw_args = (source, src_lens, ext_src, extend_vsize,
               START, END, UNK, 100)
    loss_args = (raw_articles, ext_id2word, raw_targets)
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_entity(pad, data, cuda=True):
    source_lists, targets, clusters_infos = tuple(map(list, unzip(data)))
    (cluster_lists, cluster_wpos, cluster_spos) = list(zip(*clusters_infos))

    src_nums = list(map(len, source_lists))
    cl_nums = list(map(len, cluster_lists))
    cl_nums = [cl_num if cl_num != 0 else 1 for cl_num in cl_nums]
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))
    clusters = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=4), cluster_lists)) # list of tensors, each tensor padded
    cluster_wpos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=4), cluster_wpos))
    cluster_spos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=4), cluster_spos))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (clusters, cluster_wpos, cluster_spos), cl_nums)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_ptr_hardattn(pad, data, cuda=True):
    source_lists, targets, clusters_infos, label = tuple(map(list, unzip(data)))
    (cluster_lists, cluster_wpos, cluster_spos) = list(zip(*clusters_infos))

    # print('label:', [len(_) for _ in label])
    # print(label)
    label_target = double_pad_batch_tensorize(label, pad=-1, cuda=cuda)
    label_in = double_pad_batch_tensorize(label, pad=0, cuda=cuda)
    # print('label:', label.size())
    # print('cluster:', [len(cluster) for cluster in source_lists])
    # print('targets:', [len(target) for target in targets])

    src_nums = list(map(len, source_lists))
    cl_nums = list(map(len, cluster_lists))
    cl_nums = [cl_num if cl_num != 0 else 1 for cl_num in cl_nums]
    # if label_in.size(2) != max(cl_nums) + 2:
    #     print(label)
    #     print(cl_nums)
    assert label_in.size(2) == max(cl_nums) + 2
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=5), source_lists))
    clusters = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=4), cluster_lists)) # list of tensors, each tensor padded
    cluster_wpos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=4), cluster_wpos))
    cluster_spos = list(map(pad_batch_tensorize(pad=pad, cuda=cuda, max_num=4), cluster_spos))

    # PAD is -1 (dummy extraction index) for using sequence loss
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    remove_last = lambda tgt: tgt[:-1]
    tar_in = pad_batch_tensorize(
        list(map(remove_last, targets)),
        pad=-0, cuda=cuda # use 0 here for feeding first conv sentence repr.
    )

    fw_args = (sources, src_nums, tar_in, (clusters, cluster_wpos, cluster_spos, label_in), cl_nums)
    loss_args = (target, label_target)
    return fw_args, loss_args

@curry
def convert_batch_extract_ptr_entity(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, cluster_infos = sample
        (cluster_lists, cluster_wpos, clust_spos) = cluster_infos
        id_sents = conver2id(unk, word2id, source_sents)
        id_clusters = conver2id(unk, word2id, cluster_lists)
        cluster_infos = (id_clusters, cluster_wpos, clust_spos)

        return id_sents, extracts, cluster_infos
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_entity_stop(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, cluster_infos = sample
        (cluster_lists, cluster_wpos, clust_spos) = cluster_infos
        id_sents = conver2id(unk, word2id, source_sents)
        id_clusters = conver2id(unk, word2id, cluster_lists)
        cluster_infos = (id_clusters, cluster_wpos, clust_spos)
        extracts.append(len(source_sents))
        return id_sents, extracts, cluster_infos
    batch = list(map(convert_one, batch))
    return batch

@curry
def convert_batch_extract_ptr_entity_hardattn(unk, word2id, batch):
    def convert_one(sample):
        source_sents, extracts, cluster_infos, label = sample
        (cluster_lists, cluster_wpos, clust_spos) = cluster_infos
        id_sents = conver2id(unk, word2id, source_sents)
        id_clusters = conver2id(unk, word2id, cluster_lists)
        cluster_infos = (id_clusters, cluster_wpos, clust_spos)
        extracts.append(len(source_sents))
        label.append([len(cluster_lists)+1])
        if len(label) != len(extracts):
            print('label:', label)
            print('extracts:', extracts)
        assert len(label) == len(extracts)
        # for lbss in label:
        #     for lb in lbss:
        #         if lb > len(cluster_lists) + 1:
        #             print(len(cluster_lists))
        #             print(cluster_lists)
        #             print(label)
        #             break
        return id_sents, extracts, cluster_infos, label
    batch = list(map(convert_one, batch))
    return batch

@curry
def batchify_fn_extract_ff(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums)
    loss_args = (target, )
    return fw_args, loss_args

@curry
def batchify_fn_extract_nnse(pad, data, cuda=True):
    source_lists, targets = tuple(map(list, unzip(data)))

    src_nums = list(map(len, source_lists))
    sources = list(map(pad_batch_tensorize(pad=pad, cuda=cuda), source_lists))

    tensor_type = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    #target = tensor_type(list(concat(targets)))
    target = pad_batch_tensorize(targets, pad=-1, cuda=cuda)
    loss_target = tensor_type(list(concat(targets)))

    fw_args = (sources, src_nums, target)
    loss_args = (loss_target, )
    return fw_args, loss_args


def _batch2q(loader, prepro, q, single_run=True):
    epoch = 0
    while True:
        for batch in loader:
            q.put(prepro(batch))
        if single_run:
            break
        epoch += 1
        q.put(epoch)
    q.put(None)

class BucketedGenerater(object):
    def __init__(self, loader, prepro,
                 sort_key, batchify,
                 single_run=True, queue_size=8, fork=True):
        self._loader = loader
        self._prepro = prepro
        self._sort_key = sort_key
        self._batchify = batchify
        self._single_run = single_run
        if fork:
            ctx = mp.get_context('forkserver')
            self._queue = ctx.Queue(queue_size)
        else:
            # for easier debugging
            self._queue = None
        self._process = None

    def __call__(self, batch_size: int):
        def get_batches(hyper_batch):
            indexes = list(range(0, len(hyper_batch), batch_size))
            if not self._single_run:
                # random shuffle for training batches
                random.shuffle(hyper_batch)
                random.shuffle(indexes)
            hyper_batch.sort(key=self._sort_key)
            for i in indexes:
                batch = self._batchify(hyper_batch[i:i+batch_size])
                yield batch

        if self._queue is not None:
            ctx = mp.get_context('forkserver')
            self._process = ctx.Process(
                target=_batch2q,
                args=(self._loader, self._prepro,
                      self._queue, self._single_run)
            )
            self._process.start()
            while True:
                d = self._queue.get()
                if d is None:
                    break
                if isinstance(d, int):
                    print('\nepoch {} done'.format(d))
                    continue
                yield from get_batches(d)
            self._process.join()
        else:
            i = 0
            while True:
                print('length loader:', len(self._loader))
                for batch in self._loader:
                    yield from get_batches(self._prepro(batch))
                if self._single_run:
                    break
                i += 1
                print('\nepoch {} done'.format(i), end=' ')

    def terminate(self):
        if self._process is not None:
            self._process.terminate()
            self._process.join()
