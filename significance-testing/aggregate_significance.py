import json


def aggregate_signi_fmri():
    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl_',
                  'fasttext-wiki-news_',
                  'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                  'bert-service-large', 'elmo']

    significance = {}

    with open('reports/fmri/Wilcoxon.json') as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for p in data:
                # take only results from 1000 voxels
                if emb in p and ('-1000-' in p or 'brennan' in p):
                    print(p)
                    hypotheses += 1
                    if data[p] < corrected_alpha:
                        significant += 1

            print(hypotheses)
            print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_eeg():
    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl_',
                  'fasttext-wiki-news_',
                  'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                  'bert-service-large', 'elmo']

    significance = {}

    with open('reports/eeg/Wilcoxon.json') as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for p in data:
                if emb in p:
                    hypotheses += 1
                    if data[p] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance


def aggregate_signi_gaze():
    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec',
                  'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec',
                  'bert-service-large', 'elmo']

    significance = {}

    with open('reports/gaze/Wilcoxon.json') as json_file:
        data = json.load(json_file)

        corrected_alpha = data['bonferroni_alpha']

        for emb in embeddings:
            significant = 0
            hypotheses = 0
            for p in data:
                if emb in p:
                    hypotheses += 1
                    if data[p] < corrected_alpha:
                        significant += 1

            # print(emb, significant, '/', hypotheses)
            significance[emb] = (str(significant) + '/' + str(hypotheses))

    return significance
