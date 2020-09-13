import json

all_samples = [json.loads(line) for line in open("non_match_cnndm_sample.jsonlist").readlines()]

f_w = open("non_match_cnndm.sample", 'w')
for sample in all_samples:
    cur_article = sample["article"]
    cur_summary = sample["abstract"]
    cur_article_mention = sample["filtered_input_mention_cluster"]
    cur_abstract_mention = sample["filtered_summary_mention_cluster"]
    f_w.write("<summary>\n")
    for sent in cur_summary:
        f_w.write(sent + '\n')
    f_w.write("\n")
    f_w.write("<summary_mention_cluster>\n")
    for cluster in cur_abstract_mention:
        f_w.write(str([mention["text"] for mention in cluster]) + '\n')
    f_w.write("\n")

    f_w.write("<article>\n")
    for sent in cur_article:
        f_w.write(sent + '\n')
    f_w.write("\n")
    f_w.write("<article_mention_cluster>\n")
    for cluster in cur_article_mention:
        f_w.write(str([mention["text"] for mention in cluster]) + '\n')
    f_w.write("\n") 
    f_w.write("\n")





