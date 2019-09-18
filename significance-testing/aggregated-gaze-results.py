import json
import config
import numpy as np
import matplotlib.pyplot as plt
import aggregate_significance

def extract_results_gaze(cognitive_dataset):
    result_dir = config.result_dir + 'gaze/' + cognitive_dataset + '/'
    print(result_dir)

    with open(result_dir+'options.json', 'r') as f:
        combinations = json.load(f)

        combination_results = {}
        for x,y in combinations.items():
            #print(y['feature'], y['wordEmbedding'])
            if y['feature'] not in combination_results:
                combination_results[y['feature']] = [(y['wordEmbedding'], y['AVERAGE_MSE'])]
            else:
                combination_results[y['feature']].append((y['wordEmbedding'], y['AVERAGE_MSE']))

        embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl',
                      'fasttext-wiki-news', 'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base',
                      'wordnet2vec', 'bert-service-large', 'elmo']
        baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300',
                     'random-embeddings-768', 'random-embeddings-850','random-embeddings-1024']

        avg_results = {}
        for emb_type in embeddings + baselines:
            for feat, res in combination_results.items():
                for r in res:
                    if r[0] == emb_type:
                        if not emb_type in avg_results:
                            avg_results[emb_type] = [r[1]]
                        else:
                            avg_results[emb_type].append(r[1])
            avg_results[emb_type] = sum(avg_results[emb_type]) / len(avg_results[emb_type])

        return avg_results

def extract_results_gaze_all():
    gaze_datasets = ['geco', 'zuco', 'provo', 'dundee', 'cfilt-scanpath', 'cfilt-sarcasm', 'ucl']
    combination_results = {}
    for dataset in gaze_datasets:
        result_dir = config.result_dir + 'gaze/' + dataset + '/'
        print(result_dir)

        with open(result_dir+'options.json', 'r') as f:
            combinations = json.load(f)

            for x,y in combinations.items():
                #if y['feature'] == 'nFix':
                    if y['wordEmbedding'] not in combination_results:
                        combination_results[y['wordEmbedding']] = [(y['feature'], y['AVERAGE_MSE'])]
                    else:
                        combination_results[y['wordEmbedding']].append((y['feature'], y['AVERAGE_MSE']))

        embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300', 'word2vec', 'fasttext-crawl',
                      'fasttext-wiki-news', 'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base',
                      'wordnet2vec', 'bert-service-large', 'elmo']
        baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300',
                     'random-embeddings-768', 'random-embeddings-850','random-embeddings-1024']

    avg_results = {}
    for emb_type in embeddings + baselines:
        for feat, res in combination_results.items():
            for r in res:
                print(r)
                if r[0] == emb_type:
                    if not emb_type in avg_results:
                        avg_results[emb_type] = [r[1]]
                    else:
                        avg_results[emb_type].append(r[1])

        avg_results[emb_type] = sum(avg_results[emb_type]) / len(avg_results[emb_type])

    return avg_results

def main():
    cognitive_dataset = 'ucl'
    print("Evaluating combination:", cognitive_dataset)
    results = extract_results_gaze(cognitive_dataset)

    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300',  'word2vec', 'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec', 'bert-service-large',  'elmo']
    baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-768', 'random-embeddings-850', 'random-embeddings-1024', 'random-embeddings-1024']

    significance = aggregate_significance.aggregate_signi_gaze()

    for idx, (emb, base) in enumerate(zip(embeddings, baselines)):
        avg_base = np.mean(results[base])
        avg_emb = np.mean(results[emb])
        print(emb, avg_base, avg_emb, significance[emb])


if __name__ == '__main__':
    main()
