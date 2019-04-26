from sklearn.model_selection import KFold
import numpy as np


def split_folds(input_data, outdir, folds):

    X = np.array(input_data)

    # Shuffle sentences:
    np.random.seed(123)
    np.random.shuffle(X)

    kf = KFold(n_splits=folds, shuffle=True, random_state=None)
    kf.get_n_splits(X)

    # final output format: list with a list for each fold with a list for train, dev, test
    fold_indices = []

    for train_index, test_index in kf.split(X):

        # train = 70%, dev = 10%, test = 20%
        dev = len(train_index) - int(len(test_index)/2)
        dev_index = train_index[dev:]
        new_train_index = train_index[:dev]

        fold_indices.append([new_train_index, dev_index, test_index])

    return fold_indices

def main():

    dataset = "input_data"
    folds = 5

    print("Splitting folds for ", dataset)
    split_folds(dataset, folds)


if __name__ == '__main__': main()
