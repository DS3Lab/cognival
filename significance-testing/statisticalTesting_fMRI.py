import json
import operator
import config
import os
import subprocess
import numpy as np

def extract_results(embed, baseline, result_dir):

    print(result_dir)

    for single_dir in os.listdir(result_dir):
        if not single_dir.endswith('.json'):
            if embed in single_dir and not 'random' in single_dir:
                definition = single_dir.split('_')
                defi = '_'.join(definition[:-2])
                for single_dir2 in os.listdir(result_dir ):
                    if defi in single_dir2 and baseline in single_dir2:
                        baseline_dir = single_dir2

                embeddings_file = open(result_dir + '/' + single_dir + '/' + single_dir + '.txt',
                                           'r').readlines()
                embeddings_scores = {}
                for line in embeddings_file[1:]:
                    line = line.strip().split()
                    voxels = [float(i) for i in line[1:]]
                    avg_voxels = np.mean(voxels)
                    embeddings_scores[line[0]] = avg_voxels


                baseline_file = open(result_dir + '/' + baseline_dir + '/' + baseline_dir + '.txt',
                                           'r').readlines()
                baseline_scores = {}
                for line in baseline_file[1:]:
                    line = line.strip().split()
                    voxels = [float(i) for i in line[1:]]
                    avg_voxels = np.mean(voxels)
                    baseline_scores[line[0]] = avg_voxels

                save_scores(embeddings_scores, 'embeddings_scores_'+single_dir+'.txt', baseline_scores, 'baseline_scores_'+single_dir+'.txt')
    print('Scores saved.')



def save_scores(emb_scores, emb_filename, base_scores, base_filename):
    """Save scores to temporary file. Compare embedding scores to baseline
    scores since word order and number of words differ."""

    emb_file = open(config.result_dir+'tmp/fmri/'+emb_filename, 'w')
    base_file = open(config.result_dir + 'tmp/fmri/' + base_filename, 'w')
    for word, score in emb_scores.items():
        # todo: absolute values or not?
        if word in base_scores:
            print(abs(float(score)), file=emb_file)
            print(abs(float(base_scores[word])), file=base_file)



def test_significance(baseline, model, alpha):

    command = ["python", "testSignificanceNLP/testSignificance.py", baseline, model, str(alpha), config.test]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    output, error = process.communicate()
    pvalue = float(str(output).split(": ")[-1].replace("\\n'", ""))
    model = model.split('/')[-1]
    name = model.split('.')[0]
    if "not significant" in str(output):
        print("\t\t", name, "not significant: p =", "{:10.15f}".format(pvalue))
    else:
        print("\t\t", name, "significant: p =", "{:10.15f}".format(pvalue))

    return pvalue, name




def bonferroni_correction(alpha, no_hypotheses):

    return float(float(alpha) / no_hypotheses)


def main():

    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl', 'fasttext-wiki-news',
                  'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                  'bert-service-large', 'elmo']
    baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300',
                 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300',
                 'random-embeddings-300', 'random-embeddings-768', 'random-embeddings-850', 'random-embeddings-1024',
                 'random-embeddings-1024']

    result_dir = config.result_dir + 'fmri/'
    # todo: why exclude pereira??
    gaze_datasets = ['brennan', 'wehbe', 'mitchell', 'pereira']
    for ds in gaze_datasets:
        for embed, baseline in zip(embeddings, baselines):
            print(embed, baseline)
            extract_results(embed, baseline, result_dir + ds)

    hypotheses = [1 for filename in os.listdir(config.result_dir + 'tmp/fmri/') if 'embeddings' in filename]
    print(len(hypotheses))
    alpha = bonferroni_correction(config.alpha, len(hypotheses))
    print(alpha)
    
    pvalues = {}
    pvalues['alpha'] = config.alpha
    pvalues['bonferroni_alpha'] = alpha
    report = '/Users/norahollenstein/Desktop/PhD/projects/testing/reports/fmri/' + config.test + '.json'

    for filename in os.listdir(config.result_dir + 'tmp/fmri/'):

        if 'embeddings' in filename:
            model_file = config.result_dir + 'tmp/fmri/'+filename
            baseline_file = config.result_dir + 'tmp/fmri/'+'baseline_' + filename.partition('_')[2]
            pval, name = test_significance(baseline_file, model_file, alpha)
            pvalues[name] = pval

    with open(report, 'w') as fp:
        json.dump(pvalues, fp)


if __name__ == '__main__':
    main()




