import bpy
import os
import json
import sys
from pathlib import Path

from bpy_types import Bone

from pymo.data import MocapData

print(sys.executable)
d = os.path.dirname(bpy.data.filepath)

packeges_path = '/Users/vl.korzun/.local/lib/python3.10/site-packages'

FILTERING_JOINTS = ["body_world", "b_root", "b_spine0", "b_spine1", 
                "b_spine2", "b_spine3", "b_neck0", "b_head", "b_r_shoulder",
                "b_r_arm", "b_r_arm_twist",
                "b_r_forearm", "b_r_wrist_twist",
                "b_r_wrist", "b_l_shoulder",
                "b_l_arm", "b_l_arm_twist",
                "b_l_forearm", "b_l_wrist_twist",
                "b_l_wrist", "b_r_upleg", "b_r_leg",
                "b_r_foot", "b_l_upleg", "b_l_leg", "b_l_foot"]

if d not in sys.path:
    sys.path.append(d)  
    
if packeges_path not in sys.path:
    sys.path.append(packeges_path)
    
from pymo.parsers import BVHParser
import numpy as np

def clean_scene():
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects.remove(bpy.data.objects["Armature"], do_unlink=True)
    
    
def generate_skeleton(data: MocapData):
    bpy.ops.object.armature_add(enter_editmode=False, align="WORLD", location=(0,0,0), scale=(1,1,1))
    armature = bpy.data.objects["Armature"]
    armature.select_set(True)
    
    bpy.ops.object.mode_set(mode="EDIT")
    
    bones = {}
    offsets = {}
    
    # Stage 1: calculate global offsets
    for joint in data.traverse():
        offset = np.array(data.skeleton[joint]['offsets'])
        parent = data.skeleton[joint]['parent']
        if parent is not None:
            offset += offsets[parent]
        offsets[joint] = offset

    # Stage 2: build skeleton
    for joint in data.traverse():
        if 'Nub' in joint:
            continue
        if joint == data.root_name:
            pass
        else:
            bpy.ops.armature.bone_primitive_add()
        bone = armature.data.edit_bones[-1]
        bone.head = offsets[joint]
        children = data.skeleton[joint]['children']
        parent = data.skeleton[joint]['parent']
        bone.tail = np.mean([offsets[child] for child in children], axis=0)
        if parent is not None:
            bone.parent = bones[parent]
        bone.name = joint
        bones[joint] = bone


def retarget(data, skeleton, joints_order):
    # Step 1: Build temporary Armature
    bpy.ops.object.armature_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    armature = bpy.data.objects["Armature"]
    armature.select_set(True)

    bpy.ops.object.mode_set(mode='EDIT')

    bones = {}
    for joint in joints_order:
        if 'Nub' in joint:
            continue
        if joint == joints_order[0]:  # body_world
            pass
        else:
            bpy.ops.armature.bone_primitive_add()
        bone = armature.data.edit_bones[-1]
        bone.head = np.array(data[joint])
        children = skeleton[joint]['children']
        parent = skeleton[joint]['parent']
        bone.tail = np.mean([data[child] for child in children], axis=0)
        if parent is not None:
            bone.parent = bones[parent]
        bone.name = joint
        bones[joint] = bone

    # Stage 3: retarget to metarig
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects["metarig"].select_set(True)

    bpy.ops.object.mode_set(mode='POSE')
    metarig = bpy.data.objects["metarig"]

    for bone_name in joints_order:
        metarig_bone = metarig.pose.bones[bone_name]

        constraint = metarig_bone.constraints.new('COPY_ROTATION')
        constraint.target = armature
        constraint.subtarget = bone_name
        metarig.data.bones[bone_name].select = True
        bpy.ops.pose.visual_transform_apply()

        copy_constraints = [c for c in metarig_bone.constraints if c.type == 'COPY_ROTATION']
        for c in copy_constraints:
            metarig_bone.constraints.remove(c)


data_folder = Path(d) / 'data'
sample_path = data_folder / 'val_2023_v0_000_main-agent.bvh'
pose_path = data_folder / 'json_data.json'
bvh_parser = BVHParser()
data = bvh_parser.parse(str(sample_path))
#print(data.skeleton)
clean_scene()
# generate_skeleton(data)
with open(str(pose_path)) as json_file:
    pose = json.load(json_file)

bpy.data.scenes['Scene'].frame_set(0)
retarget(pose, skeleton=data.skeleton, joints_order=list(data.traverse()))
bpy.ops.anim.keyframe_insert_menu(type='Rotation')
