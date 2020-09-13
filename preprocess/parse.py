from pycorenlp import StanfordCoreNLP
import pickle
import re
import json

nlp = StanfordCoreNLP('http://localhost:9000')


def corenlp_process(sample):
    """process sample with StanforCoreNLP, input: str"""
    output = nlp.annotate(sample, properties={
              'annotators': 'tokenize,ssplit,coref,entitymentions',
              'outputFormat': 'json',
              'timeout': '50000'
              })
    return output


#data = pickle.load(open("../nyt_wo_author_seg/nyt-wo-author_seg1.pkl", 'rb'))

#id_s = 580000
#id_e = 600000

data = [json.loads(line) for line in open("../nyt_wo_author_seg/" + str(id_s) + "_" + str(id_e) + ".jsonlist").readlines()]

f_w_data = open(str(id_s) + "_" + str(id_e) + "_parsed.jsonlist", 'w')
f_w_error = open(str(id_s) + "_" + str(id_e) + "_error.ids", 'w')

index = 1
for cur_dict in data:
    index += 1
    if index % 200 == 0:
        f_w_data.flush()
    cur_id = cur_dict["id"]
    print(cur_id)
    cur_input = cur_dict["input"].replace("</title>","").replace("</content>",".").replace("\n"," ").strip()
    cur_summa = ".".join(re.split("[;]", cur_dict["summary"]))
    # process
    cur_input_process = corenlp_process(cur_input)
    cur_summa_process = corenlp_process(cur_summa)
    cur_dict["summary_parsed"] = cur_summa_process
    cur_dict["input_parsed"] = cur_input_process
    if isinstance(cur_summa_process, dict) == False or isinstance(cur_input_process, dict) == False:
        f_w_error.write(str(cur_id) + '\n')   
    else:
        print(cur_summa_process.keys(), cur_input_process.keys()) 
        f_w_data.write(json.dumps(cur_dict) + '\n')






