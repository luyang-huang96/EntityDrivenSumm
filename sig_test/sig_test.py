from art import aggregators
from art import scores
from art import significance_tests
import os

def compare_oracle_vs_noracle():
    system = 'flat_ctx'
    test = significance_tests.ApproximateRandomizationTest(
           scores.Scores.from_file(open('meteor_scores/aligned/noracle_' + system + '_wp.txt')), 
           scores.Scores.from_file(open('meteor_scores/aligned/full_' + system + '_wp.txt')), 
           aggregators.average)
    sig_level = test.run()
    print(system)
    print(sig_level)


def compare_all_pairwise():
    upper = "output_file/human/process/"
    all_files = os.listdir(upper)
    #print(all_files)
    r1_files = [elem for elem in all_files if ".coh" in elem]
    r2_files = [elem for elem in all_files if ".entityRep" in elem]
    rl_files = [elem for elem in all_files if ".gramm" in elem]
    r4_files = [elem for elem in all_files if ".info" in elem]

    for cur_files in [r1_files, r2_files, rl_files, r4_files]:
        print(cur_files)
        for i in range(len(cur_files) - 1):
            for j in range(i + 1, len(cur_files)):
                s1 = upper + cur_files[i]
                s2 = upper + cur_files[j]
                if s1 == s2: continue
                compare_systems(s1, s2)
        print("======================================")

def compare_systems(system_1, system_2):
    test = significance_tests.ApproximateRandomizationTest(
           scores.Scores.from_file(open(system_1)), 
           scores.Scores.from_file(open(system_2)), 
           aggregators.average)
    sig_level = test.run()
    print('system1: %s\tsystem2: %s\t' % (system_1, system_2))
    print(sig_level)
    print("")

"""
#compare_all_pairwise()
upper_folder = "output_file/rouge/oracle/processed/"
file_ours = "ours_4_new.rouge"
file_bl = "ours_4_new.rouge"
"""
upper_folder = "/data2/luyang/process-cnn/CameraReadyResults/rouge_results/rouge/"
file_1 = 'pointnet+entity+rl'
#file_2 = "joint+absrouge+0.01coh.rougel"

files_2 = ['dca_m7',
    'bottomup',
    'fastrl_output',
    'pgc_output']
appends = ['.rouge1', '.rouge2', '.rougel']
for cur_file in files_2:
    for _app in appends:
        compare_systems(upper_folder + 'rouge_scores_' + file_1 + _app, upper_folder + 'rouge_scores_'  + cur_file + _app)
#compare_all_pairwise()


