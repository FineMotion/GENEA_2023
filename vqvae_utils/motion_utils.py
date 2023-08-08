from .pymo_vqvae.parsers import BVHParser
from .pymo_vqvae.preprocessing import MocapParameterizer, ConstantsRemover, Numpyfier, RootTransformer, JointSelector
from sklearn.pipeline import Pipeline
from typing import List
from pymo.data import MocapData
import numpy as np

FPS: int = 30

joints: List[str] = ['body_world', 'spine', 'shoulder',
                     'arm', 'wrist', 'forearm', 'head', 'neck0',
                     'spine3', 'spine2', 'spine1', 'spine0']


def load_bvh_file(file_path: str) -> MocapData:
    """
    Loads a BVH file.

    Args:
        file_path (str): Path to the BVH file.

    Returns:
        MocapData: Parsed motion capture data.
    """
    parser = BVHParser()
    parsed_data = parser.parse(file_path)
    return parsed_data


def create_data_pipeline() -> Pipeline:
    """
    Creates a pipeline for processing motion capture data.

    Returns:
        Pipeline: The data processing pipeline.
    """
    return Pipeline([
        ('root', RootTransformer('hip_centric')),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('jtsel', JointSelector(joints, include_root=True)),
        ('np', Numpyfier())
    ])


def process_data_pipeline(parsed_data: MocapData, pipeline: Pipeline) -> np.ndarray:
    """
    Processes motion capture data using a pipeline.

    Args:
        parsed_data (MocapData): The parsed motion capture data.
        pipeline (Pipeline): The data processing pipeline.

    Returns:
        np.ndarray: Processed data samples.
    """

    processed_samples = pipeline.fit_transform([parsed_data])
    return processed_samples[0]


def split_bvh_into_blocks(processed_data: np.ndarray, beats: np.ndarray) -> List[np.ndarray]:
    """
    Splits BVH data into blocks based on beat intervals.

    Args:
        processed_data (np.ndarray): The processed motion capture data.
        beats (np.ndarray): List of beat frame indices.

    Returns:
        List[np.ndarray]: BVH data blocks.
    """

    blocks = []
    frame_beats = beats

    for i in range(len(frame_beats) - 1):
        start = frame_beats[i]
        end = frame_beats[i + 1]

        assert start != end, "Start and end frame indices must be different"

        block = processed_data[start:end]
        blocks.append(block)

    return blocks
