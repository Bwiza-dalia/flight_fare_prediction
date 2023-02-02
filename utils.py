import pandas as pd


def get_encoded_label(text, column, dataset_unlabeled: pd.DataFrame, dataset_labeled: pd.DataFrame):
    obj = dict()
    label = ''
    if column == 'class':
        if text == 'Economy':
            return 1
        else:
            return 0

    # get unique values from unlabeled
    unlabeled = list(set(dataset_unlabeled[column]))

    # get unique values from labeled
    labeled = list(set(dataset_labeled[column]))

    # create a dictionary of labeled and unlabeled
    for index in range(len(unlabeled)):
        obj[unlabeled[index]] = labeled[index]

    # return labeled value of text
    if text in obj.keys():
        label = obj[text]
    else:
        label = None
    return label
