"""Configuration of experiment, datasets and models"""

# result directory
result_dir = '/Users/norahollenstein/Desktop/PhD/projects/testing/results/'

# recording modalities: gaze, eeg and/or fmri
modalities = ['fmri']

# eye-tracking: ['geco', 'zuco', 'provo', 'dundee', 'cfilt-scanpath', 'cfilt-sarcasm', 'ucl']
# eeg: ['n400', 'ucl', 'zuco', 'naturalspeech']
# fmri: ['brennan', 'wehbe', 'mitchell', 'pereira']

datasets = {'gaze': ['geco', 'zuco', 'provo', 'dundee', 'cfilt-scanpath', 'cfilt-sarcasm', 'ucl'], 'eeg': ['n400', 'ucl', 'zuco', 'naturalspeech'], 'fmri': ['brennan', 'wehbe', 'mitchell', 'pereira']}

# todo: needed?
cognitive_dataset = 'all'

# feature to analyze
feature = 'nRegres_to'

# baseline: random or any embedding type
baseline = 'random'

# which embeddings to evaluate
embeddings = 'elmo'

# testing parameters
alpha = '0.01'
test = 'Wilcoxon'  # or Permutation, Wilcoxon
