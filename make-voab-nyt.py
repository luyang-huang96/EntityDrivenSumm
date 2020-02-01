import glob
import json, pickle, os, time
import collections
from cytoolz import concat

if __name__ == '__main__':
    #finished_files_dir = '/data2/luyang/process-nyt/finished_files/'
    finished_files_dir = '/data2/luyang/process-cnn/finished_files/'
    files = glob.glob(finished_files_dir + 'train/*')
    vocab_counter = collections.Counter()
    cl_words = []
    id = 0
    start = time.time()
    for file in files:
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except:
                print('error file')
                print(file)
                continue
            raw_article = data['article']
            abss = data['abstract']
            input_cluster = data['filtered_input_mention_cluster']
            cnt = [_.lower().split(' ') for _ in raw_article]
            abs = [_.lower().split(' ') for _ in abss]
            art_tokens = list(concat(cnt))
            abs_tokens = list(concat(abs))
            tokens = art_tokens + abs_tokens
            tokens = [t.strip() for t in tokens]  # strip
            tokens = [t for t in tokens if t != ""]  # remove empty
            vocab_counter.update(tokens)
            for cluster in input_cluster:
                for mention in cluster:
                    cl_words.extend(mention['text'].lower().split(' '))
            id += 1
            if id % 10000 == 0:
                print('{} finished'.format(id))
                print(time.time() - start)

    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pickle.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file")
    print(sorted(vocab_counter.items(), key=lambda x:x[1], reverse=True)[:50000])
    vocab = dict(sorted(vocab_counter.items(), key=lambda x:x[1], reverse=True)[:50000])
    # info
    count0 = 0
    count_all = 0
    stopWords = set()
    with open('/home/luyang/stopwords.txt', 'r', encoding='utf-8') as f:
        for line in f:
            stopWords.add(line.strip())
    for word in cl_words:
        if word in vocab and word not in stopWords:
            count0 += 1
        if word not in stopWords:
            count_all += 1
    print(count0, count_all)


