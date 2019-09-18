# CogniVal
### A framework for cognitive word embedding evaluation

This repository contains the code for all experiments in the following paper:  
Nora Hollenstein, Antonio de la Torre, Ce Zhang & Nicolas Langer. CogniVal: A Framework for Cognitive Word Embedding Evaluation. _CoNLL_ (2019).


## Regression models

The following set of scripts generates and fits a neural network model to predict cognitive data such as fMRI, eye-tracking 
and EEG from word embedding inputs.

This process is divided into 4 different handlers to facilitate code reading, debugging and modularity:

- The ``dataHandler.py`` receives the input data used to fit, predict and validate the model. It takes care of parsing, joining and dividing it into training and
testing sets.

- The ``fileHandler.py`` organizes the different configurations, and handles the reading and writing into them. 

- The ``modelHandler.py`` is the core of the project, where the models are generated, fitted and prediction is done.

- Finally, the ``plotHandler.py`` comes into play to generate visually understandable results.


There are two main ways to run the scripts:

1. If using ``script.py`` arguments can be direclty passed to run the model.
 In case no arguments are passed, a command line interface will start and ask for the necessary inputs. These are:
- word embedding
- cognitive data
- feature of the cognitive data

In the cases of fMRI and EEG, we want to predict a vector representing the data and not a specific feature. Thus the cli will 
not ask for a particular feature to be predicted.    

In order for the script to run properly the necessary information has to be previously stored inside the setupConfig.json 
file. This information consists of the names of the datafiles, the path to where they are stored, the number of hidden layers and nodes for the neural network
etc.

Example_1:

We want to run: "dundee" with the feature "First_fix_dur" and the word embedding "glove-50".

Command to run:

``python script.py path/to/setupConfig.json -c dundee  -f First_fix_dur  -w glove-50``

An example of the ``setupConfig.json`` with the necessary information to run this case is stored in ``config/example_1.json``.

In ``example_1.json`` we have the necessary information to run dundee with any of its features and two different word embeddings: glove-50 and word2vec. Further word embeddings 
or cognitive datasets can be added in a similar fashion. 

In case of big word embeddings like word2vec, the datafiles are chunked into several pieces to avoid a MemoryError. This has 
to be performed separately using the chunker method inside of dataHandler.

2. The second way of running the script is to pass a ``controller.json`` to the ``scriptController.py``.

This will in turn have the same effect as ``script.py``, however multiple combinations and models can be run in parallel. An example of
a ``controllerConfig.json`` is found inside ``config/``


## Significance testing

To run the statistical significance tests as described in the paper, place your result files in `significance-testing/results/`.
We use the implementation of the Wilcoxon test for NLP provided by Dror et al. (2018).

Then you can set the specific configuration in `significance-testing/config.py` and run these scripts in the following order:
1. `statisticalTesting.py`  
This will create and test all test results for the chosen modalities in `significance-testing/reports/`.
2. `aggregated-eeg-results.py` or fMRI or eye-tracking (scripts differ slightly)  
This will output how many of your hypotheses are accepted under the Bonferroni correction (see paper for detailed description)

