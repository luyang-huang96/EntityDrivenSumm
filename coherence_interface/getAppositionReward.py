# encoding=utf8
import numpy as np
import nltk

NN_tags = set(['NN','NNS','NNP','NNPS'])
NNP_tags = set(['NNP','NNPS'])
Wh_tags = set(['WDT','WP','WP$','WRB'])

def makePairs(comma_ind_list):
    comma_ind_pairs = set()
    for i in range(len(comma_ind_list)-1):
        comma_ind_pairs.add((comma_ind_list[i],comma_ind_list[i+1]))
    return comma_ind_pairs

def getAppositionReward(tok_sents):

    sent_score = 0
    for tok_sent in tok_sents:
        tok_sent = tok_sent.split(" ")
        tok_sent = [tok for tok in tok_sent if tok != '']
        # find if 2 commas
        comma_ind = np.where(np.array(tok_sent) == ",")[0]
        if len(comma_ind) >= 2:

            comma_ind_pairs = makePairs(comma_ind)
            tok_sent = [x for x in tok_sent if x != '']
            tagged = nltk.pos_tag(tok_sent)

            for comma_ind_pair in comma_ind_pairs:
                comma1 = comma_ind_pair[0]
                comma2 = comma_ind_pair[1]
                # check if atleast one other token before two commas and last comma not the last token
                if comma1 < comma2-1 and comma2 < len(tagged)-1:
                    bf_comma1 = tagged[comma1-1][1]
                    af_comma1 = tagged[comma1+1][1]
                    #  if Wh-word right after comma - penalize since its relative clause
                    if af_comma1 in Wh_tags:
                        sent_score -= 1
                    else:
                        # if Noun right before the first comma and not a proper noun right after the  first comma
                        # then penalize since apposition
                        if bf_comma1 in NN_tags and af_comma1 not in NNP_tags:
                            sent_score -= 1
    return sent_score


if __name__ == "__main__":
	"""
	Input : list of sentences in a summary
    Output: score (value <= 0)
	It checks for the presence of Relative Clause or Apposition
	Rules:
	1. If two(or more) commas present
	2. For each pair of adjacent commas:
		i. atleast one token between the commas and second(last) comma not the last token
		ii. if token after first comma a Wh POS tag - then score = -1
		iii. if token before first comma a Noun and after first comma not a proper noun - then score = -1
	"""

	print(getAppositionReward(["although , who friendship somewhat healed years later it was a devastating loss to croly . He is a friend"]))
	print(getAppositionReward(["kubler , who retired from cycling in 1957 , remained a revered figure in , the wealthy alpine nation ."]))
	print(getAppositionReward(["kubler , who retired from cycling in 1957 , remained a revered figure in the wealthy alpine nation ."]))
	print(getAppositionReward(["My friend , a girl , who loves icecream , has come to live with me . "]))
