import bpy
import os
import json
import sys
from pathlib import Path


d = os.path.dirname(bpy.data.filepath)
# path to site-packages with pandas
packeges_path = os.environ.get("BLENDER_PACKAGES")

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
    
if packeges_path is not None and packeges_path not in sys.path:
    sys.path.append(packeges_path)
  
from pymo.parsers import BVHParser
from pymo.data import MocapData
import numpy as np

def clean_scene():
    bpy.ops.object.mode_set(mode='OBJECT')
    for obj in bpy.data.objects:
        if obj.name == "Armature":
            print("Deleting Armature...")
            bpy.data.objects.remove(bpy.data.objects["Armature"], do_unlink=True)
    
    
def generate_skeleton(data: MocapData):
    print("Generatig Skeleton...")
    bpy.ops.object.armature_add(enter_editmode=False, align="WORLD", location=(0,0,0), scale=(1,1,1))
    armature = bpy.data.objects["Armature"]
    armature.name = "metarig"
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
        tail = np.mean([offsets[child] for child in children], axis=0)
        bone.tail = tail
        if bone.length < 1e-6:
            bone.tail[1] += 0.1
            
        if parent is not None:
            bone.parent = bones[parent]
        bone.name = joint
        bones[joint] = bone
    print(bones)


def retarget(data, skeleton, joints_order, do_retarget=True):
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
        tail = np.mean([data[child] for child in children], axis=0)
        bone.tail = tail
#        if 'twist'
#        if tail == bone.head:
#            tail[1] += 0.1
#        if bone.length < 1e-12:
#            # it is foot twist
#            # move a little in parent direction
#            parent_head = bones[parent].head
#            direction = bone.head - parent_head
#            direction = direction /  np.linalg.norm(direction)
#            bone.tail += direction * 0.1 
        
        if parent is not None:
            bone.parent = bones[parent]
        bone.name = joint
        bones[joint] = bone
    
    if not do_retarget:
        return
    
    # Stage 3: retarget to metarig
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.data.objects["metarig"].select_set(True)

    bpy.ops.object.mode_set(mode='POSE')
    metarig = bpy.data.objects["metarig"]

    for bn in metarig.pose.bones:
        bone_name = bn.name
#        if 'Nub' in joint:
#            continue
        bone = metarig.pose.bones[bone_name]

        constraint = bone.constraints.new('COPY_ROTATION')
        constraint.target = armature
        constraint.subtarget = bone_name
        metarig.data.bones[bone_name].select = True
        bpy.ops.pose.visual_transform_apply()

        copy_constraints = [c for c in bone.constraints if c.type == 'COPY_ROTATION']
        for c in copy_constraints:
            bone.constraints.remove(c)


data_folder = Path(d) / 'data'
sample_path = data_folder / 'val_2023_v0_000_main-agent.bvh' # input bvh-file
pose_folder = data_folder / 'val_000_frames' # folder with extracted positions by frames via Python script
bvh_parser = BVHParser()
data = bvh_parser.parse(str(sample_path))
#print(data.skeleton)

# GENERATING REST POSE
generate_skeleton(data)

# BUILD POSES AND TRANSFER ROTATIONS  
for i in range(10): # set number of frames to transfer
    bpy.data.scenes['Scene'].frame_set(i)
    clean_scene()
    
    pose_path = pose_folder / f'frame_{i:04d}.json'
    with open(str(pose_path)) as json_file:
        pose = json.load(json_file)
    
    retarget(pose, skeleton=data.skeleton, joints_order=list(data.traverse()))
    bpy.ops.anim.keyframe_insert_menu(type='Rotation')


# BUILD FIRST POSE TO COMPARE
with open(str(pose_folder / f'frame_{0:04d}.json')) as json_file:
    pose = json.load(json_file)
    clean_scene()
    retarget(pose, skeleton=data.skeleton, joints_order=list(data.traverse()), do_retarget=False)