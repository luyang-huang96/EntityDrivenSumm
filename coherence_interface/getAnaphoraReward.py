# encoding=utf8
import numpy as np
import nltk

NN_tags = set(['NN','NNS','NNP','NNPS'])
APN_tags = set(['PRP','PRP$'])

def getAnaphoraReward(tok_sents):

    sents = " ".join(tok_sents).split(" ")
    sents = [tok for tok in sents if tok != '']
    tags = [tag[1] for tag in nltk.pos_tag(sents)]

	# get the first occurance of a pronoun
    first_pronoun_ind = -1
    for i,tag in enumerate(tags):
        if tag in APN_tags:
            first_pronoun_ind = i
            break
	# if no prounoun return 0
    if first_pronoun_ind == -1:
        return 0

	# get the first occurance of a noun
    first_noun_ind = -1
    for i,tag in enumerate(tags):
        if tag in NN_tags:
            first_noun_ind = i
            break
    # if first occurance of pronoun happened before the noun return -1 else 0
    if first_pronoun_ind < first_noun_ind:
        return -1
    else:
        return 0

if __name__ == "__main__":
	"""
	Input : list of sentences in a summary
    Output: score (value is either 0 or -1)
	It checks whether first occcurance of pronoun is before the first occurance of noun
	Rules:
	1. If no pronoun return 0
	2. If first occurance of pronoun is before the first occurance of noun return -1
	3. else return -1
	"""
	print(getAnaphoraReward(["donald trump is the president of united states of america .", "he is a bad president ."]))
	print(getAnaphoraReward(["he is a bad president .", "donald trump is the president of united states of america ."]))
	print(getAnaphoraReward(["he is a bad president since donald trump is stupid .", "donald trump is the president of united states of america ."]))
	print(getAnaphoraReward(["his daughter is stupid since donald trump is stupid .", "he is the president of united states of america ."]))
	print(getAnaphoraReward(["donald trump 's daughter is stupid since he is stupid .", "he is the president of united states of america ."]))
	print(getAnaphoraReward(["Mary said he likes icecream . "]))
	print(getAnaphoraReward(["they reported heavy shooting in the orange county.", "but the orange county media couldn 't get any specifics ."]))
