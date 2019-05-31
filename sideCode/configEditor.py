import json

path = "/home/delatvan/current/cognitive-embedding-evaluation/config/"
configFile = 'setupConfig.json'

with open(path+configFile,'r') as fR:
	setup = json.load(fR)

bert_service_base = {
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

bert_service_large = {
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
            	}


for elem in setup["cogDataConfig"]:
	setup["cogDataConfig"][elem]["wordEmbSpecifics"]["glove-50"] = {
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
                        ],
                        [
							30	
                        ],
                        [
                        	20
                        ],
                        [
                        	5
                        ]
                    ],
                    "validation_split": 0.2
                }

	setup["cogDataConfig"][elem]["wordEmbSpecifics"]["bert-service-base"] = bert_service_base
	setup["cogDataConfig"][elem]["wordEmbSpecifics"]["bert-service-large"]=bert_service_large

with open(configFile,'w') as fW:
	json.dump(setup,fW,indent=4,sort_keys = True)