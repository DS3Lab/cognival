import json

cd = 'mitchell'
dim = 1000
start = 0
end = 2
wordEmbeddings = ["word2vec","fasttext-crawl","fasttext-crawl-subword","fasttext-wiki-news", "fasttext-wiki-news-subword",
    "elmo","bert-base","bert-large","bert-service-base","bert-service-large","glove-50","glove-100","glove-200","glove-300",
    "random-embeddings-100","random-embeddings-1024","random-embeddings-200","random-embeddings-300",
    "random-embeddings-50","random-embeddings-768","random-embeddings-850","wordnet2vec"]

dic1 = {'cognitiveData':{}}
for i in range(start,end+1):
	dic1['cognitiveData'][cd+'-'+str(dim)+'-'+str(i)] = {
        "features": [
          "ALL_DIM"
        ]
      }

dic1['wordEmbeddings'] = wordEmbeddings
dic1["configFile"] = "config/setupConfig.json"

print(dic1)

with open(cd+'-'+str(dim)+'.json','w') as fW:
	json.dump(dic1,fW, indent=4, sort_keys = True)