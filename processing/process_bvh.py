from pymo.parsers import BVHParser
from pymo.preprocessing import MocapParameterizer, ConstantsRemover, Numpyfier, RootTransformer, JointSelector
from sklearn.pipeline import Pipeline
import numpy as np
import joblib as jl
import os

FPS = 30

joints = ['body_world', 'spine', 'shoulder',
          'arm', 'wrist', 'forearm', 'head', 'neck0',
          'spine3', 'spine2', 'spine1', 'spine0']


def load_bvh_file(file_path):
    parser = BVHParser()
    parsed_data = parser.parse(file_path)
    return parsed_data


def data_pipline(parsed_data):
    data_pipe = Pipeline([
        ('root', RootTransformer('hip_centric')),
        ('exp', MocapParameterizer('expmap')),
        ('cnst', ConstantsRemover()),
        ('jtsel', JointSelector(joints, include_root=True)),
        ('np', Numpyfier())
    ])

    processed_samples = data_pipe.fit_transform([parsed_data])

    jl.dump(data_pipe, 'pipeline.sav')
    return processed_samples[0]


def split_bvh_into_blocks(processed_data, beats):
    blocks = []
    #frame_beats = np.array(beats / frame_time).astype(int)
    frame_beats = beats
    #print((beats / frame_time))
    for i in range(len(frame_beats) - 1):
        start = frame_beats[i]
        end = frame_beats[i + 1]

        assert start != end

        block = processed_data[start:end]

        blocks.append(block)
    return blocks
