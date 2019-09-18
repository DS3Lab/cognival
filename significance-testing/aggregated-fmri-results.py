import json
import config
import matplotlib.pyplot as plt
import aggregate_significance

def extract_results(cognitive_dataset):
    if not cognitive_dataset == "all":
        result_dir = config.result_dir + 'fmri/' + cognitive_dataset + '/'
        print(result_dir)

        with open(result_dir+'options.json', 'r') as f:
            combinations = json.load(f)

            combination_results = {}
            for x,y in combinations.items():
                #print(y['feature'], y['wordEmbedding'])
                    if y['wordEmbedding'] not in combination_results:
                        combination_results[y['wordEmbedding']] = [y['AVERAGE_MSE']]
                    else:
                        combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])

            #average over subjects:
            avg_results = {}
            for emb, res in combination_results.items():
                avg_results[emb] = sum(res) / len(res)

            return avg_results

    else:
        fmri_datasets = ['brennan', 'wehbe', 'mitchell', 'pereira']
        combination_results = {}
        for dataset in fmri_datasets:
            result_dir = config.result_dir + 'fmri/' + dataset + '/'
            print(result_dir)

            with open(result_dir + 'options.json', 'r') as f:
                combinations = json.load(f)

                for x, y in combinations.items():
                    if '-1000-' in y['cognitiveData'] or 'brennan' in y['cognitiveData']:
                        #print('only 100 voxels')
                    # print(y['feature'], y['wordEmbedding'])
                        if y['wordEmbedding'] not in combination_results:
                            combination_results[y['wordEmbedding']] = [y['AVERAGE_MSE']]
                        else:
                            combination_results[y['wordEmbedding']].append(y['AVERAGE_MSE'])

        avg_results = {}
        for emb, res in combination_results.items():
            avg_results[emb] = sum(res) / len(res)

        return avg_results



def main():
    cognitive_dataset = 'wehbe'
    print("Evaluating combination:", )
    results = extract_results(cognitive_dataset)

    embeddings = ['glove-50', 'glove-100', 'glove-200', 'glove-300',  'word2vec', 'fasttext-crawl-subword', 'fasttext-wiki-news-subword', 'bert-service-base', 'wordnet2vec', 'bert-service-large', 'elmo']
    baselines = ['random-embeddings-50', 'random-embeddings-100', 'random-embeddings-200', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-300', 'random-embeddings-768', 'random-embeddings-850', 'random-embeddings-1024', 'random-embeddings-1024']

    fig, ax = plt.subplots(1,len(embeddings), sharey=True)
    significance = aggregate_significance.aggregate_signi_fmri()

    for idx, (emb, base) in enumerate(zip(embeddings, baselines)):
        plt.xticks(rotation=90)
        avg_base = results[base]
        avg_emb = results[emb]
        print(emb, avg_base, avg_emb, significance[emb])
        ax[idx].bar([0, 1], [avg_base, avg_emb], align='center', tick_label=['baseline', emb], color=['b', 'g'])
        ax[idx].set_xticklabels(['baseline', 'x'], rotation='vertical')

        #for tick in ax[idx].get_xticklabels():
        #    tick.set_rotation(90)

    plt.show()


if __name__ == '__main__':
    main()
