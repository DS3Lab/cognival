"""Configuration of experiment, datasets and models"""

# result directory
result_dir = './results/'

# directory to save testing reports
report_dir = './reports/'

# recording modalities: gaze, eeg and/or fmri
modality = 'gaze'

# eye-tracking: ['geco', 'zuco', 'provo', 'dundee', 'cfilt-scanpath', 'cfilt-sarcasm', 'ucl']
# eeg: ['n400', 'ucl', 'zuco', 'naturalspeech']
# fmri: ['brennan', 'wehbe', 'mitchell', 'pereira']

datasets = {'gaze': ['geco', 'zuco', 'provo', 'dundee', 'cfilt-scanpath', 'cfilt-sarcasm', 'ucl'], 'eeg': ['n400', 'ucl', 'zuco', 'naturalspeech'], 'fmri': ['brennan', 'wehbe', 'mitchell', 'pereira']}

# testing parameters
alpha = 0.01
test = 'Wilcoxon'  # or Permutation, Wilcoxon
