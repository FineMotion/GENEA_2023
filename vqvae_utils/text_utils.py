import numpy as np
import pandas as pd

SAMPLE_RATE = 44100
FPS = 30
HOP_LENGTH = SAMPLE_RATE // FPS


def load_tsv_file(file_path: str) -> pd.DataFrame:
    """
    Load a TSV file.

    Args:
        file_path (str): Path to the TSV file.

    Returns:
        pd.DataFrame: The loaded TSV data in a pandas DataFrame.
    """

    tsv_data = pd.read_csv(file_path, sep='\t', header=None, names=['start', 'end', 'word'])
    return tsv_data


def get_borders(tsv_data: pd.DataFrame, pause_threshold: float = 1.0) -> np.ndarray:
    """
    Determine the borders of phrases in the TSV data based on pause duration.

    Args:
        tsv_data (pd.DataFrame): The TSV data in a pandas DataFrame.
        pause_threshold (float, optional): The pause duration to use as the threshold for phrase breaks (defaults 1.0)

    Returns:
        np.ndarray: An array with the start and end times of each phrase.
    """

    if len(tsv_data) == 0:
        return np.array([])
    tsv_data['prev_end'] = tsv_data['end'].shift(1)
    tsv_data['pause'] = tsv_data['start'] - tsv_data['prev_end']
    tsv_data.loc[tsv_data.index[0], 'pause'] = 0

    tsv_data['phrase'] = (tsv_data['pause'] > pause_threshold).cumsum()
    phrases = tsv_data.groupby('phrase').agg({'start': 'first', 'end': 'last'})

    return np.array(phrases)


