# encoding=utf8
import numpy as np
import nltk
import re
from collections import defaultdict,Counter
from collections import OrderedDict
regex = r"(?:\w+ (?:NN|NNS|NNP|NNPS)) ?(?:(?:\w+ (?:NN|NNS|NNP|NNPS))*)"

def getChunks(tok_sent):

    tags = nltk.pos_tag(tok_sent)
    new_sent = " ".join([" ".join(tag) for tag in tags])
    raw_chunks = re.findall(regex,new_sent)
    np_chunks = [' '.join(chunk.replace("N",'').replace("P",'').replace("S",'').strip().split()) for chunk in raw_chunks]
    return np_chunks

def getGrid(np_list):

    # get unique entities
    entity_grid = OrderedDict()
    for sent_ents in np_list:
        entity_grid.update([(x,[]) for x in sent_ents if x != ""])
    # create grid
    for sent_ents in np_list:
        sent_ents_cnt = Counter(sent_ents)
        for ent in entity_grid.keys():
            if sent_ents_cnt.get(ent) is None:
                entity_grid[ent].append(0)
            else:
                entity_grid[ent].append(sent_ents_cnt[ent])
    return entity_grid

def entityRepetition(tok_sents):
    np_list = []
    for tok_sent in tok_sents:
        tok_sent = tok_sent.split(" ")
        tok_sent = [x for x in tok_sent if x != '']
        np_list.append(getChunks(tok_sent))

    egrid = getGrid(np_list)
    if len(egrid) == 0:
        return 0
    sent_score = 0
#     print(egrid)
    for k,v in egrid.items():
        if len(k.split(" ")) >= 2:
            if sum(v) > 1:
                sent_score -= np.mean(v)
    return sent_score/len(egrid)

if __name__ == "__main__":
	"""
	Input : list of sentences in a summary
    Output: score (value <= 0)
	It penalizes noun chunk longer than one word is mentioned more than once in an article
	Rules:
	1. Get all the recursive nouns as noun chunks
	2. If the noun chunk is greater two words and occurs in more than one sentence in the summary penalize it with number of times it is mentioned, else return 0. If no noun chunks return 0
	"""
	print(entityRepetition(["donald trump is the president of united states of america ."]))
	print(entityRepetition(["donald trump is the president of united states of america .", "he is a bad president ."]))
	print(entityRepetition(["donald trump is the president of united states of america .", "donald trump is a bad president ."]))
	print(entityRepetition(["donald trump is the president of united states of america with milania trump his daughter .", "donald trump is a bad president .", "donald trump and milania trump are stupid ."]))

