import json
from os import listdir
from os.path import isfile, join

name = 'pereira'
dim = 1000

dataset = 'cognitive-data/fmri/'+name+'_data/aggregated/'

mypath = '/home/delatvan/Dropbox/university/ETH/4fs/projektArbeit/datasets/'+dataset

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f)) and 'scaled_'+str(dim)+'_' in f]
ord_list = sorted(onlyfiles)


# example = {
#             "dataset": "cognitive-data/eeg/n400/n400_scaled.txt",
#             "features": [
#                 "ALL_DIM"
#             ],
#             "type": "multivariate_output",
#             "wordEmbSpecifics": {
#                 "bert-base": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             400
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "bert-large": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             600
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "elmo": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             600
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "fasttext-crawl": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "fasttext-crawl-subword": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "fasttext-wiki-news": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "fasttext-wiki-news-subword": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "glove-100": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         30
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             30
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "glove-200": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             100
#                         ],
#                         [
#                             50
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "glove-300": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "glove-50": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             26
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-100": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         30
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             30
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-1024": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             600
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-200": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             100
#                         ],
#                         [
#                             50
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-300": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-50": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             26
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-768": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             400
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "random-embeddings-850": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             400
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "word2vec": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             50
#                         ],
#                         [
#                             150
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 },
#                 "wordnet2vec": {
#                     "activations": [
#                         "relu"
#                     ],
#                     "batch_size": [
#                         128
#                     ],
#                     "cv_split": 3,
#                     "epochs": [
#                         100
#                     ],
#                     "layers": [
#                         [
#                             400
#                         ],
#                         [
#                             200
#                         ]
#                     ],
#                     "validation_split": 0.2
#                 }
#             }
#         }

# conf = {}

setupConfig = 'cognitive-embedding-evaluation/config/setupConfig.json'
with open(setupConfig,'r') as fr:
    setup = json.load(fr)

print(ord_list)
# print(len(onlyfiles))

for i, elem in enumerate(ord_list):
	setup['cogDataConfig'][name+'-'+str(dim)+'-'+str(i)] = {
            "dataset": dataset+elem,
            "features": [
                "ALL_DIM"
            ],
            "type": "multivariate_output",
            "wordEmbSpecifics": {
                "bert-base": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            400
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                },
                "bert-large": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            600
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                },
                "elmo": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            600
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                },
                "fasttext-crawl": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "fasttext-crawl-subword": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "fasttext-wiki-news": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "fasttext-wiki-news-subword": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "glove-100": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        30
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            30
                        ]
                    ],
                    "validation_split": 0.2
                },
                "glove-200": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            100
                        ],
                        [
                            50
                        ]
                    ],
                    "validation_split": 0.2
                },
                "glove-300": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "glove-50": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            26
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-100": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        30
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            30
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-1024": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            600
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-200": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            100
                        ],
                        [
                            50
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-300": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-50": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            26
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-768": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            400
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                },
                "random-embeddings-850": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            400
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                },
                "word2vec": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            50
                        ],
                        [
                            150
                        ]
                    ],
                    "validation_split": 0.2
                },
                "wordnet2vec": {
                    "activations": [
                        "relu"
                    ],
                    "batch_size": [
                        128
                    ],
                    "cv_split": 3,
                    "epochs": [
                        100
                    ],
                    "layers": [
                        [
                            400
                        ],
                        [
                            200
                        ]
                    ],
                    "validation_split": 0.2
                }
            }
        }

with open(setupConfig,'w') as fW:
	json.dump(setup,fW, indent=4, sort_keys=True)