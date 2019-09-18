import json
import config
import os
import numpy as np
from testing_helpers import bonferroni_correction, test_significance, save_scores


def extract_results(modality, embed, baseline, result_dir):
    print(result_dir)

    if modality == 'fmri':

        for single_dir in os.listdir(result_dir):
            if not single_dir.endswith('.json'):
                if embed in single_dir and not 'random' in single_dir:
                    definition = single_dir.split('_')
                    defi = '_'.join(definition[:-2])
                    for single_dir2 in os.listdir(result_dir):
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

                    save_scores(embeddings_scores, 'embeddings_scores_' + single_dir + '.txt', baseline_scores,
                                'baseline_scores_' + single_dir + '.txt')

    if modality == 'eeg':

        for single_dir in os.listdir(result_dir):
            if not single_dir.endswith('.json'):
                if embed in single_dir and not 'random' in single_dir:
                    definition = single_dir.split('_')
                    defi = '_'.join(definition[:-2])
                    for single_dir2 in os.listdir(result_dir):
                        if defi in single_dir2 and baseline in single_dir2:
                            baseline_dir = single_dir2

                    embeddings_file = open(result_dir + '/' + single_dir + '/' + single_dir + '.txt',
                                           'r').readlines()

                    embeddings_scores = {}
                    for line in embeddings_file[1:]:
                        line = line.strip().split()
                        electrodes = [float(i) for i in line[1:]]
                        avg_electrodes = np.mean(electrodes)
                        embeddings_scores[line[0]] = avg_electrodes

                    baseline_file = open(result_dir + '/' + baseline_dir + '/' + baseline_dir + '.txt',
                                         'r').readlines()
                    baseline_scores = {}
                    for line in baseline_file[1:]:
                        line = line.strip().split()
                        electrodes = [float(i) for i in line[1:]]
                        avg_electrodes = np.mean(electrodes)
                        baseline_scores[line[0]] = avg_electrodes

                    save_scores(embeddings_scores, 'embeddings_scores_' + single_dir + '.txt', baseline_scores,
                                'baseline_scores_' + single_dir + '.txt')

    if modality == 'gaze':
        for single_dir in os.listdir(result_dir):
            if not single_dir.endswith('.json'):
                if embed in single_dir and not 'random' in single_dir:
                    definition = single_dir.split('_')
                    defi = '_'.join(definition[:-2])
                    for single_dir2 in os.listdir(result_dir):
                        if defi in single_dir2 and baseline in single_dir2:
                            baseline_dir = single_dir2
                        else:
                            continue

                    try:
                        embeddings_file = open(result_dir + '/' + single_dir + '/' + single_dir + '.txt',
                                               'r').readlines()
                    except FileNotFoundError:
                        alt_name = definition[0] + '-gaze' + '_' + definition[1] + '_' + definition[2] + '_' + \
                                   definition[3]
                        embeddings_file = open(result_dir + '/' + single_dir + '/' + alt_name + '.txt',
                                               'r').readlines()
                    embeddings_scores = dict(line.strip().split() for line in embeddings_file[1:])

                    baseline_file = open(result_dir + '/' + baseline_dir + '/' + baseline_dir + '.txt',
                                         'r').readlines()
                    baseline_scores = dict(line.strip().split() for line in baseline_file[1:])

                    save_scores(embeddings_scores, 'embeddings_scores_' + single_dir + '.txt', baseline_scores,
                            'baseline_scores_' + single_dir + '.txt')

    print('Scores saved.')



def main():
    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl', 'fasttext-wiki-news',
                  'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                  'bert-service-large', 'elmo']
    baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300',
                 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300',
                 'random-embeddings-300', 'random-embeddings-768', 'random-embeddings-850', 'random-embeddings-1024',
                 'random-embeddings-1024']

    result_dir = config.result_dir + config.modality + '/'
    datasets = config.datasets[config.modality]
    for ds in datasets:
        for embed, baseline in zip(embeddings, baselines):
            extract_results(config.modality, embed, baseline, result_dir + ds)

    hypotheses = [1 for filename in os.listdir(config.result_dir + 'tmp/' + config.modality + '/') if 'embeddings_' in filename]
    print(len(hypotheses))
    alpha = bonferroni_correction(config.alpha, len(hypotheses))
    print(alpha)

    pvalues = {'alpha': config.alpha, 'bonferroni_alpha': alpha}
    report = config.report_dir + config.modality + '/' + config.test + '.json'

    for filename in os.listdir(config.result_dir + 'tmp/' + config.modality + '/'):

        if 'embeddings_' in filename:
            model_file = config.result_dir + 'tmp/' + config.modality + '/' + filename
            baseline_file = config.result_dir + 'tmp/' + config.modality + '/' + 'baseline_' + filename.partition('_')[2]
            pval, name = test_significance(baseline_file, model_file, alpha)
            pvalues[name] = pval

    with open(report, 'w') as fp:
        json.dump(pvalues, fp)


if __name__ == '__main__':
    main()
