import argparse

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix


def main(args):
    submission_df = pd.read_csv(args.submission_csv)
    test_df = pd.read_csv(args.test_csv)

    print(submission_df.columns, test_df.columns)
    print(submission_df.shape, test_df.shape)
    pred = submission_df[args.target_colunm].values
    gt = test_df[args.target_colunm].values

    if submission_df.shape[0] != test_df.shape[0]:
        raise ValueError('Submission and test set have different number of samples')

    print(f'Accuracy: {accuracy_score(gt, pred)}')
    print(f'Confusion matrix: \n {confusion_matrix(gt, pred)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--submission_csv', default='homeworks/hw1/data/submission.csv')
    parser.add_argument('--test_csv', default='homeworks/hw1/data/test_gt.csv')
    parser.add_argument('--target_colunm', default='order0')

    args = parser.parse_args()
    main(args)
