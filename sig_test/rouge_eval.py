import pyrouge
import os
import sys
#from itertools import izip
from itertools import zip_longest as izip
import json
import tempfile


all_tagged = [
"joint+0.5r2+0.5rl.jsonlist",
"joint+0.5r2+0.5rl+anaphora.jsonlist",
"joint+0.5r2+0.5rl+apposition.jsonlist",
"joint+all+new.jsonlist",
"joint+coh1.jsonlist",
#"joint+coh2.jsonlist",
'joint+noentity.jsonlist'
]


tmp_path = "/data2/luyang/junck/tmp2"
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
tempfile.tempdir = tmp_path

# upper_path = "/data2/luyang/process-cnn/CameraReadyResults/js/"
# write_path = "/data2/luyang/process-cnn/CameraReadyResults/rouge_results/"
upper_path = "/data2/luyang/process-nyt/CameraReadyResults/js/"
write_path = "/data2/luyang/process-nyt/CameraReadyResults/rouge_results/"

#upper_path = "../summary_model_output/sota/cnndm/rouge/"
#write_path = "../summary_model_output/sota/cnndm/rouge/"

base_dir = "/data2/luyang/process-nyt/CameraReadyResults/rouge_results/" # path to where you want to store result
gold_tagged = upper_path + "human_summary.jsonlist"  # path to gold summaries

#tagged = "pointnet+entity+rl.jsonlist"
#tagged = "pointnet.jsonlist"
#tagged = "pointnet+entity.jsonlist"
#tagged = "pointnet+entity+absml.jsonlist"
#tagged = "pgc_output.josnlist"
#tagged = "joint+allrewards.jsonlist"
tagged = "joint+0.5r2+0.5rl.jsonlist"


for tagged in all_tagged:
    r1 = 0
    r2 = 0
    rl = 0
    count = 0
    mach_tagged = upper_path + tagged # path to machine summaries

    tmp_gold = base_dir + "/" + tagged.replace(".jsonlist", "") + "/tmp_gold_seq"
    tmp_dec = base_dir + "/" + tagged.replace(".jsonlist", "") + "/tmp_dec_seq"
    if not os.path.exists(tmp_gold):
        os.makedirs(tmp_gold)
    if not os.path.exists(tmp_dec):
        os.makedirs(tmp_dec)

    t = 0
    rouge_scores = []
    with open(gold_tagged) as gold, open(mach_tagged) as mach:
        for g,m in izip(gold,mach):
            #if t == 10: break
            #m = [sent for sent in m.split("\n")] #  get sent tokens
            #g = [sent for sent in g.split("\n")] # get sent tokens
            json_m = json.loads(m)
            json_g = json.loads(g)
            m = json_m["summary"]
            g = json_g["original"]
            id_m = "N/A"
            id_g = "N/A"
            #id_m = json_m["id"]
            #id_g = json_g["id"]
            #print("=====================")
            #print("cur id: ", id_m, id_g)
            #print("=====================")
            with open(os.path.join(tmp_gold,"gold."+str(t)+".txt"),'w') as fout:
                fout.writelines("\n".join(g))
            with open(os.path.join(tmp_dec,"mach."+str(t)+".txt"),'w') as fout:
                fout.write("\n".join(m))

            ref_dir = tmp_gold
            dec_dir = tmp_dec

            r = pyrouge.Rouge155('/home/luyang/ROUGE-1.5.5')
            r.model_filename_pattern = 'gold.#ID#.txt'
            r.system_filename_pattern = 'mach.(\d+).txt'
            r.model_dir = ref_dir
            r.system_dir = dec_dir


            try:
                rouge_results = r.convert_and_evaluate()
                output_dict = r.output_to_dict(rouge_results)
                rouge1 = output_dict["rouge_1_f_score"]
                rouge2 = output_dict["rouge_2_f_score"]
                rougel = output_dict["rouge_l_f_score"]
                r1 += rouge1
                r2 += rouge2
                rl += rougel
                count += 1

                rouge_scores.append({"id":[id_m, id_g], "score": "%6.4f %6.4f %6.4f" % (rouge1,rouge2,rougel)})
                #print("current score: ", rouge_scores[-1])
            except:
                rouge_scores.append({"id":[id_m, id_g], "score": "NA"})
                pass

            os.remove(os.path.join(tmp_gold,"gold."+str(t)+".txt"))
            os.remove(os.path.join(tmp_dec,"mach."+str(t)+".txt"))
            t+=1
    print('average of {}. r1: {}, r2:{}, rl:{}'.format(tagged, r1/count, r2/count, rl/count))



    with open(write_path + "rouge_scores_" + tagged,'w') as fout:
        for elem in rouge_scores:
            fout.write(json.dumps(elem) + '\n')
