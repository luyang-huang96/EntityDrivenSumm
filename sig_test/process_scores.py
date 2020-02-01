import json, os, re



OPEN_DIR = '/data2/luyang/process-cnn/CameraReadyResults/rouge_results/'
WRITE_DIR = '/data2/luyang/process-cnn/CameraReadyResults/rouge_results/rouge/'

if not os.path.exists(WRITE_DIR):
    os.makedirs(WRITE_DIR)

if __name__ == '__main__':
    files = ['dca_m7.jsonlist',
    'bottomup.jsonlist',
    'fastrl_output.jsonlist',
    'pgc_output.jsonlist',
    'pointnet+entity+rl.jsonlist']
    for file in files:
        file = OPEN_DIR + 'rouge_scores_' + file
        print(file)
        r1s = [float(json.loads(line)["score"].split(" ")[0]) if json.loads(line)["score"].split(" ")[0] != 'NA' else 0.0 for line in open(file).readlines()]
        r2s = [float(json.loads(line)["score"].split(" ")[1]) if json.loads(line)["score"].split(" ")[0] != 'NA' else 0.0 for line in open(file).readlines()]
        rls = [float(json.loads(line)["score"].split(" ")[2]) if json.loads(line)["score"].split(" ")[0] != 'NA' else 0.0 for line in open(file).readlines()]
        print(len(r1s))
        filename_base = file.split('/')[-1].replace('.jsonlist','')
        with open(WRITE_DIR + filename_base + '.rouge1', 'w') as f:
            for r1 in r1s:
                f.write(str(r1) + '\n')
        with open(WRITE_DIR + filename_base + '.rouge2', 'w') as f:
            for r2 in r2s:
                f.write(str(r2) + '\n')
        with open(WRITE_DIR + filename_base + '.rougel', 'w') as f:
            for rl in rls:
                f.write(str(rl) + '\n')
