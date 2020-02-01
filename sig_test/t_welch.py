from __future__ import division
from collections import namedtuple
import json
import numpy as np
import scipy.stats as st

TtestResults = namedtuple("Ttest", "T p")

def t_welch(x, y, tails=2):
    """Welch's t-test for two unequal-size samples, not assuming equal variances
    """
    assert tails in (1,2), "invalid: tails must be 1 or 2, found %s"%str(tails)
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = x.size, y.size
    vx, vy = x.var(), y.var()
    df = int((vx/nx + vy/ny)**2 / # Welch-Satterthwaite equation
        ((vx/nx)**2 / (nx - 1) + (vy/ny)**2 / (ny - 1)))
    t_obs = (x.mean() - y.mean()) / np.sqrt(vx/nx + vy/ny)
    p_value = tails * st.t.sf(abs(t_obs), df)
    return TtestResults(t_obs, p_value)



def t_test(path1, path2):
    score1 = [eval(json.loads(line)["score"].split(" ")[1]) for line in open(path1).readlines()]
    score2 = [eval(json.loads(line)["score"].split(" ")[1]) for line in open(path2).readlines()]
    
    score1_proc = [elem for elem in score1 if elem != "NA"]
    score2_proc = [elem for elem in score2 if elem != "NA"]
    print("file1: {}, file2: {}".format(path1, path2))
    print(t_welch(score1_proc, score2_proc, tails=2))
    print(np.mean(score1_proc), np.mean(score2_proc))



def main():
    upper_path = '/data2/luyang/process-cnn/CameraReadyResults/js/'
    #upper_path = ""    
    """
    # for nyt
    files1 = ["joint+absrouge+0.01coh_model.jsonlist"]
    files2 = ["joint+allrewards_model.jsonlist", "joint+absrouge+0.005apposition_model.jsonlist",
              "joint+absrouge+0.005anaphora_model.jsonlist", "paulus_2_model.jsonlist",
              "fastrl_model.jsonlist", "joint+extml+absrl+noentity_model.jsonlist",
              "pointnet+entity+rl_model.jsonlist", "joint+absrouge_model.jsonlist"]
    """
    # for cnndm
    files1 = ["joint+0.5r2+0.5rl.jsonlist", "joint+0.5r2+0.5rl+apposition.jsonlist"]
    files2 = [
    'bottomup.jsonlist',
    'dca_m7.jsonlist',
    'fastrl_output.jsonlist',
    'pgc_output.jsonlist']

    

    for file1 in files1:
        for file2 in files2:
            path1 = upper_path + 'rouge_scores_' + file1
            path2 = upper_path + 'rouge_scores_' + file2
            t_test(path1, path2)
            print("")    


if __name__ == "__main__":
    main()






