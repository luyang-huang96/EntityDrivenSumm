import scipy.stats
import json
import numpy as np

# read data
all_corre_score = []

all_model_score = []
all_human_score = []

all_results = [json.loads(line) for line in open("nyt_summ_sample_result.jsonlist").readlines()]
for sample in all_results:
    if 'summary_middle' in sample:
        cur_result = []
        cur_result.append(np.mean(sample["summary_pos_score"]))
        cur_result.append(np.mean(sample["summary_middle_score"]))
        cur_result.append(np.mean(sample["summary_neg_score"]))
        comp_score = [3, 2, 1]
        #all_corre_score.append(scipy.stats.stats.spearmanr(cur_result, comp_score))
        #print("model: ", cur_result)
        #print("compa: ", comp_score)
        #print("correlation: ", scipy.stats.stats.spearmanr(cur_result, comp_score))
        all_model_score += cur_result
        all_human_score += comp_score
    else:
        cur_result = []
        cur_result.append(np.mean(sample["summary_pos_score"]))
        cur_result.append(np.mean(sample["summary_neg_score"]))
        comp_score = [3, 1]
        #all_corre_score.append(scipy.stats.stats.spearmanr(cur_result, comp_score))
        #print("model: ", cur_result)
        #print("compa: ", comp_score)
        #print("correlation: ", scipy.stats.stats.spearmanr(cur_result, comp_score))
        all_model_score += cur_result
        all_human_score += comp_score


print("final results:")
assert len(all_model_score) == len(all_human_score)
print(len(all_model_score))
print("overall correlation: ", scipy.stats.stats.spearmanr(all_model_score, all_human_score))
#print("mean: ", np.mean(all_corre_score))
#print("median: ", np.median(all_corre_score))



