import numpy as np
import pandas as pd


def load_tsv_file(file_path):
    tsv_data = pd.read_csv(file_path, sep='\t', header=None, names=['start', 'end', 'word'])
    return tsv_data


def get_borders(tsv_data, pause_threshold=1.0):
    if len(tsv_data) == 0:
        return np.array([])
    tsv_data['prev_end'] = tsv_data['end'].shift(1)
    tsv_data['pause'] = tsv_data['start'] - tsv_data['prev_end']
    tsv_data.loc[tsv_data.index[0], 'pause'] = 0

    tsv_data['phrase'] = (tsv_data['pause'] > pause_threshold).cumsum()
    phrases = tsv_data.groupby('phrase').agg({'start': 'first', 'end': 'last'})

    return np.array(phrases)
