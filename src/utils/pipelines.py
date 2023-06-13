from pymo.preprocessing import MocapParameterizer, Numpyfier, JointSelector, ConstantsRemover, ChannelSelector
from sklearn.pipeline import Pipeline


def position_pipeline():
    data_pipe = Pipeline([
            ('pos', MocapParameterizer('position')),
            ('jtsel', JointSelector([
                "b_root", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_neck0", "b_head", "b_r_shoulder",
                "b_r_arm", "b_r_arm_twist",
                "b_r_forearm", "b_r_wrist_twist",
                "b_r_wrist", "b_l_shoulder",
                "b_l_arm", "b_l_arm_twist",
                "b_l_forearm", "b_l_wrist_twist",
                "b_l_wrist", "b_r_upleg", "b_r_leg",
                "b_r_foot", "b_l_upleg", "b_l_leg", "b_l_foot"
            ], include_root=True)),
            ('np', Numpyfier())
        ])
    return data_pipe


def ortho6d_pipeline():
    data_pipe = Pipeline([
        ('jtsel', JointSelector(["b_root", "b_spine0", "b_spine1", "b_spine2", "b_spine3", "b_neck0", "b_head",
                                 "b_r_shoulder", "b_r_arm", "b_r_arm_twist",
                                 "b_r_forearm", "b_r_wrist_twist",
                                 "b_r_wrist", "b_l_shoulder",
                                 "b_l_arm", "b_l_arm_twist",
                                 "b_l_forearm", "b_l_wrist_twist",
                                 "b_l_wrist", "b_r_upleg", "b_r_leg",
                                 "b_r_foot", "b_l_upleg", "b_l_leg", "b_l_foot"], include_root=True)),
        ('ortho6d', MocapParameterizer('ortho6d')),
        ('clsel', ChannelSelector(["a1x", "a1y", "a1z", "a2x", "a2y", "a2z"], filter_root=False)),
        ('np', Numpyfier()),
    ])
    return data_pipe


def ortho6d_full_pipeline():
    data_pipe = Pipeline([
        ('ortho6d', MocapParameterizer('ortho6d')),
        ('clsel', ChannelSelector(["a1x", "a1y", "a1z", "a2x", "a2y", "a2z"], filter_root=False)),
        ('np', Numpyfier())
    ])
    return data_pipe
