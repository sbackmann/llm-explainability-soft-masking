import numpy as np
import pandas as pd
import os

def generics_kb_df(data_save_dir, with_labels=False, display=False):
    path_to_data = os.path.join(data_save_dir, "generics")
    os.makedirs(path_to_data, exist_ok=True)
    file = os.path.join(path_to_data, "generics_kb_large.csv")
    if not os.path.isfile(file):
        raise ValueError("The dataset generics large is not found")
    with open(file) as csv_file:
        read_csv = pd.read_csv(csv_file, delimiter=",")
    data = read_csv['sentence_without_last_token']
    if with_labels:
        y = read_csv['sentence'].apply(lambda x: x.split(' ')[-1].split('.')[0])
        return data, y
    else:
        return data, None

def generics_kb(data_save_dir, with_labels=False, display=False):
    data, y = generics_kb_df(data_save_dir, with_labels, display)
    return np.array(data), np.array(y)
    '''
    path_to_data = os.path.join(data_save_dir, "generics")
    os.makedirs(path_to_data, exist_ok=True)
    file = os.path.join(path_to_data, "generics_kb_large.csv")
    if not os.path.isfile(file):
        raise ValueError("The dataset generics large is not found")
    csv_file = open(file)
    read_csv = pd.read_csv(csv_file, delimiter=",")
    data = np.array(list(read_csv['sentence_without_last_token']))
    print(f"Loading Generics Large dataset: {data.shape[0]}")
    if with_labels:
        sentence = list(read_csv['sentence'])
        y = np.array([s.split(' ')[-1].split('.')[0] for s in list(sentence)])
        return data, y
    else:
        return data, None'''