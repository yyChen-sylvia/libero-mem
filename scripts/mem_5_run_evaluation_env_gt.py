"""
Replay and evaluate LIBERO demonstrations in simulation, recording robot
trajectories, object-centric annotations, and subgoal completion statistics,
then save all results into a consolidated pickle file (all_trajs_<eval_mode>.pkl).

This can be used for later evaluation.

Usage examples:
    python mem_5_run_evaluation_env_gt.py \
        --libero_task_suite libero_mem \
        --libero_raw_data_dir ./demonstration_data/libero_mem \
        --libero_result_dir ./libero_mem_eval \
        --eval_mode seen \
        --eval_runs 20

    python mem_5_run_evaluation_env_gt.py \
        --libero_task_suite libero_mem \
        --libero_raw_data_dir ./demonstration_data/libero_mem \
        --libero_result_dir ./libero_mem_eval \
        --eval_mode unseen \
        --eval_runs 20
"""


import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import robosuite.utils.transform_utils as T
from tqdm import tqdm
from libero.libero import benchmark

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)

import sys
sys.path.append('../')

IMAGE_RESOLUTION = 256

class NoOpChecker:
    def __init__(self, max_gripper_time=15):
        self.gripper_time = max_gripper_time
        self.max_gripper_time = max_gripper_time

    def filter_noops(self, orig_actions):
        prev_action = None
        actions = []
        for action in orig_actions:
            if self.is_noop(action, prev_action):
                continue
            prev_action = action
            actions.append(action)

        for _ in range(10):
            actions.append(np.array([0,0,0.01,0,0,0,-1]))

        return actions

    def is_noop(self, action, prev_action=None, threshold=1e-4):
        """
        Returns whether an action is a no-op action.

        A no-op action satisfies two criteria:
            (1) All action dimensions, except for the last one (gripper action), are near zero.
            (2) The gripper action is equal to the previous timestep's gripper action.

        Explanation of (2):
            Naively filtering out actions with just criterion (1) is not good because you will
            remove actions where the robot is staying still but opening/closing its gripper.
            So you also need to consider the current state (by checking the previous timestep's
            gripper action as a proxy) to determine whether the action really is a no-op.
        """
        # Special case: Previous action is None if this is the first action in the episode
        # Then we only care about criterion (1)
        if prev_action is None:
            return np.linalg.norm(action[:-1]) < threshold

        # Normal case: Check both criteria (1) and (2)
        gripper_action = action[-1]
        prev_gripper_action = prev_action[-1]
        self.gripper_time += 1
        if gripper_action != prev_gripper_action:
            self.gripper_time = 0 # just changed the gripper state
        return np.linalg.norm(action[:-1]) < threshold and self.gripper_time >= self.max_gripper_time

from enum import Enum

class Flags(Enum):
    IS_SAVING_TRAJECTORY = 1
    IS_REPEATING_TRAJECTORY = 2
    IS_SKIPPING_TRAJECTORY = 3
    IS_RELABELING_TRAJECTORY = 4

import numpy as np
import cv2

def get_task_nouns(task_description):
    if task_description in TASK_NOUN_DICT:
        return TASK_NOUN_DICT[task_description]
    # 1/0
    return []

def min_max_normalize(image, old_min=0, old_max=1, new_min=0, new_max=1):
    """
    Normalize an image using Min-Max normalization.

    Parameters:
        image (numpy array): Input image.
        new_min (float): Minimum value of the normalized range.
        new_max (float): Maximum value of the normalized range.

    Returns:
        normalized_image (numpy array): Normalized image.
    """
    # Convert the image to float for processing
    image = image.astype(np.float32)

    # Get min and max of the original image
    old_min = np.min(image)
    old_max = np.max(image)
    # if DEBUG_MODE:
    #     print(old_min, old_max)
    # Apply Min-Max normalization formula
    normalized_image = (image - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

    return (normalized_image*255).astype(np.uint8)

def get_visual(obs, key, normalize_depth=True):
    """
    Processes a visual observation (image, segmentation, or depth) from obs.

    Args:
        obs (dict): The observation dictionary from the environment.
        key (str): The key to retrieve the image/segmentation/depth.
        normalize_depth (bool): Whether to normalize depth images automatically.

    Returns:
        np.ndarray: The processed visual output.
    """
    # Fetch and flip horizontally
    visual = obs[key][:, ::-1]

    # Depth maps need normalization
    if 'depth' in key:
        if 'agentview' in key:
            visual = min_max_normalize(visual, old_min=0.97, old_max=1.0)
        elif 'eye_in_hand' in key:
            visual = min_max_normalize(visual, old_min=0.71, old_max=1.0)
        else:
            if normalize_depth:
                visual = min_max_normalize(visual)  # fallback

    # Segmentations need to be uint8
    if 'segmentation' in key:
        visual = visual.astype(np.uint8)

    # Rotate 180 degrees
    visual = cv2.rotate(visual, cv2.ROTATE_180)

    return visual

color4label = {}
def get_color4label(label):
    if label in color4label:
        return color4label[label]
    else:
        new_color = np.random.randint(0, 255, size=(3, ))
        color4label[label] = tuple(new_color.astype(np.int32))
        return color4label[label]


class TrajectoryRecorder:
    def __init__(self, image_resolution=128):
        self.image_resolution = image_resolution

    def reset_trackstate(self):
        # Storage
        self.states = []
        self.actions = []
        self.ee_states = []
        self.gripper_states = []
        self.joint_states = []
        self.robot_states = []
        self.agentview_images = []
        self.eye_in_hand_images = []
        self.agentview_depths = []
        self.eye_in_hand_depths = []
        self.agentview_segs = []
        self.eye_in_hand_segs = []
        self.agentview_boxes = []
        self.eye_in_hand_boxes = []
        self.cur_obs = None
        self.cur_subgoal = '0'
        self.oc_satisfied_subgoals = {}
        self.subgoal_object_counter = {}
        self.subgoal_object_relevant_THRESHOLD = 20
        for noun in self.nouns:
            label = self.remapper[noun]
            self.oc_satisfied_subgoals[label] = '0'
            self.subgoal_object_counter[label] = 0

    def reset_object_mappers(self, env):
        ##########################################
        # """ Functions to handle utilities """"
        def simplify(text):
            for key, value in SIMPLIFYING_DICT.items():
                if key in text:
                    return text.replace(key, value)
            return text

        def revise_remapper(remapper, double_filter=True):
            for key, value in remapper.items():
                value = value.split(' ')
                value = ' '.join(value)

                if double_filter:
                    value = simplify(value)

                remapper[key] = value

            return remapper

        def get_manual_geomid(geom_name2id, geom_names=['']):
            ids = []
            for name in geom_names:
                ids.append(geom_name2id[name])
            return ids

        def not_match_any(geom_id, selected_ids):
            if geom_id is None:
                return False
            for id in selected_ids:
                if id in geom_id:
                    return False
            return True
        ##########################################

        nouns = ['gripper0']
        remapper = {
            'gripper0': 'robot'
        }
        
        # Retrieve the mapping from geometry IDs to names
        geom_name2id = {env.sim.model.geom_id2name(geom_id): geom_id  
                        for geom_id in range(env.sim.model.ngeom)}
        geom_ids = list(geom_name2id.keys())

        # Retrieve the mapping from each noun to geom id
        id_list = {}
        for noun in nouns:
            re_noun = noun.replace(' ', '_')
            id_list[re_noun] = []
            for geom in geom_ids:
                # print(geom)
                if geom is not None and re_noun in geom:
                    if re_noun == 'book':
                        if geom not in ['office_book_shelf', 'black_book_1', 'black_book_2']:
                            id_list[re_noun].append(geom_name2id[geom])
                    elif re_noun == 'shelf':
                        if geom not in ['office_book_shelf']:
                            id_list[re_noun].append(geom_name2id[geom])
                    else:
                        id_list[re_noun].append(geom_name2id[geom])
        id_list['gripper0'] = []
        for geom in geom_ids:
            # print(geom)
            if geom is not None and 'gripper0' in geom:
                id_list['gripper0'].append(geom_name2id[geom])

        if 'wooden_two_layer_shelf_1_g0' in geom_name2id:
            keywords = ['cabinet', 'cabinet shelf']
            for keyword in keywords:
                if keyword in id_list and len(id_list[keyword]) == 0:
                    id_list[keyword] = get_manual_geomid(geom_name2id, geom_names=['wooden_two_layer_shelf_1_g0', 'wooden_two_layer_shelf_1_g1', 'wooden_two_layer_shelf_1_g2', 'wooden_two_layer_shelf_1_g3', 'wooden_two_layer_shelf_1_g4', 'wooden_two_layer_shelf_1_g5', 'wooden_two_layer_shelf_1_g6'])

        if 'wooden_cabinet_1_g0' in geom_name2id:
            if 'top_drawer' in id_list:
                id_list['top_drawer'] = get_manual_geomid(geom_name2id, geom_names=['wooden_cabinet_1_g6', 'wooden_cabinet_1_g7', 'wooden_cabinet_1_g8', 'wooden_cabinet_1_g9', 'wooden_cabinet_1_g10', 'wooden_cabinet_1_g11', 'wooden_cabinet_1_g12', 'wooden_cabinet_1_g13', 'wooden_cabinet_1_g14', 'wooden_cabinet_1_g15', 'wooden_cabinet_1_g16', 'wooden_cabinet_1_g17', 'wooden_cabinet_1_g18'])
            if 'middle_drawer' in id_list:
                id_list['middle_drawer'] = get_manual_geomid(geom_name2id, geom_names=['wooden_cabinet_1_g19', 'wooden_cabinet_1_g20', 'wooden_cabinet_1_g21', 'wooden_cabinet_1_g22', 'wooden_cabinet_1_g23', 'wooden_cabinet_1_g24', 'wooden_cabinet_1_g25', 'wooden_cabinet_1_g26', 'wooden_cabinet_1_g27', 'wooden_cabinet_1_g28', 'wooden_cabinet_1_g29', 'wooden_cabinet_1_g30', 'wooden_cabinet_1_g31'])
            if 'bottom_drawer' in id_list:
                id_list['bottom_drawer'] = get_manual_geomid(geom_name2id, geom_names=['wooden_cabinet_1_g32', 'wooden_cabinet_1_g33', 'wooden_cabinet_1_g34', 'wooden_cabinet_1_g35', 'wooden_cabinet_1_g36', 'wooden_cabinet_1_g37', 'wooden_cabinet_1_g38', 'wooden_cabinet_1_g39', 'wooden_cabinet_1_g40', 'wooden_cabinet_1_g41', 'wooden_cabinet_1_g42'])
        elif 'white_cabinet_1_g0' in geom_name2id:
            if 'top_drawer' in id_list:
                id_list['top_drawer'] = get_manual_geomid(geom_name2id, geom_names=['white_cabinet_1_g6', 'white_cabinet_1_g7', 'white_cabinet_1_g8', 'white_cabinet_1_g9', 'white_cabinet_1_g10', 'white_cabinet_1_g11', 'white_cabinet_1_g12', 'white_cabinet_1_g13', 'white_cabinet_1_g14', 'white_cabinet_1_g15', 'white_cabinet_1_g16', 'white_cabinet_1_g17', 'white_cabinet_1_g18'])
            if 'middle_drawer' in id_list:
                id_list['middle_drawer'] = get_manual_geomid(geom_name2id, geom_names=['white_cabinet_1_g19', 'white_cabinet_1_g20', 'white_cabinet_1_g21', 'white_cabinet_1_g22', 'white_cabinet_1_g23', 'white_cabinet_1_g24', 'white_cabinet_1_g25', 'white_cabinet_1_g26', 'white_cabinet_1_g27', 'white_cabinet_1_g28', 'white_cabinet_1_g29', 'white_cabinet_1_g30', 'white_cabinet_1_g31'])
            if 'bottom_drawer' in id_list:
                id_list['bottom_drawer'] = get_manual_geomid(geom_name2id, geom_names=['white_cabinet_1_g32', 'white_cabinet_1_g33', 'white_cabinet_1_g34', 'white_cabinet_1_g35', 'white_cabinet_1_g36', 'white_cabinet_1_g37', 'white_cabinet_1_g38', 'white_cabinet_1_g39', 'white_cabinet_1_g40', 'white_cabinet_1_g41', 'white_cabinet_1_g42'])
        
        # if 'desk_caddy_1_g0' in geom_name2id: # doesnt workp
        #     if 'front_compartment' in id_list:
        #         id_list['front_compartment'] = get_manual_geomid(geom_name2id, geom_names=['desk_caddy_1_g1'])
        #     if 'left_compartment' in id_list:
        #         id_list['left_compartment'] = get_manual_geomid(geom_name2id, geom_names=['desk_caddy_1_g0'])
        #     if 'right_compartment' in id_list:
        #         id_list['right_compartment'] = get_manual_geomid(geom_name2id, geom_names=['desk_caddy_1_g4', 'desk_caddy_1_g5'])

        remain_geom_ids = [geom_id for geom_id in geom_ids if not_match_any(geom_id, id_list.keys())]
        remain_obj_keys = [geom_id[:-3] for geom_id in remain_geom_ids if '_g0' in geom_id and 'robot0' not in geom_id]

        for key in remain_obj_keys:
            if 'chefmate_8_frypan' in key:
                remapper[key] = 'frying pan'
            else:
                remapper[key] = key.replace('_',' ')
            re_noun = key
            id_list[re_noun] = []
            for geom in geom_ids:
                # print(geom)
                if geom is not None and re_noun in geom:
                    if re_noun == 'black_book_1' and geom == 'black_book_1':
                        continue
                    id_list[re_noun].append(geom_name2id[geom])
        nouns+= remain_obj_keys

        remapping_values = list(remapper.values()) # assert that all verbs belong to this remapper
        self.remapper = revise_remapper(remapper)
        self.remapping_values = remapping_values
        self.id_list = id_list
        self.nouns = nouns

    def record(self, obs, action, seg_to_mask_fn, get_bbox_fn):
        """
        Record one step worth of data
        """

        # Save action
        self.actions.append(action)

        # Save gripper, joint, and end-effector state
        if "robot0_gripper_qpos" in obs:
            self.gripper_states.append(obs["robot0_gripper_qpos"])
        self.joint_states.append(obs["robot0_joint_pos"])
        self.ee_states.append(
            np.hstack((
                obs["robot0_eef_pos"],
                T.quat2axisangle(obs["robot0_eef_quat"]),
            ))
        )
        self.robot_states.append(
            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
        )

        self.cur_obs = obs

        # Save rotated images
        self.agentview_images.append(get_visual(obs, "agentview_image"))
        self.eye_in_hand_images.append(get_visual(obs, "robot0_eye_in_hand_image"))

        # Process segmentations
        ego_objs, exo_objs, ego_seg, exo_seg = self.process_segmentations(
            obs, seg_to_mask_fn, get_bbox_fn
        )

        self.eye_in_hand_boxes.append(ego_objs)
        ego_image = get_visual(obs, "robot0_eye_in_hand_image")
        for label, data in ego_objs.items():
            ind, bbox = data
            color = get_color4label(label)
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

            x_min, y_min, w, h = bbox
            x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
            x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
            cv2.rectangle(ego_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
            text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
            cv2.putText(ego_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)
        ego_image_boxed = ego_image

        self.agentview_boxes.append(exo_objs)
        exo_image = get_visual(obs, "agentview_image")
        for label, data in exo_objs.items():
            ind, bbox = data
            color = get_color4label(label)
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

            x_min, y_min, w, h = bbox
            x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
            x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
            cv2.rectangle(exo_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
            text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
            cv2.putText(exo_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)
        exo_image_boxed = exo_image

        self.eye_in_hand_segs.append(ego_seg)
        self.agentview_segs.append(exo_seg)

        # Save depths
        exo_depth = get_visual(obs, "agentview_depth")
        ego_depth = get_visual(obs, "robot0_eye_in_hand_depth")
        self.agentview_depths.append(exo_depth)
        self.eye_in_hand_depths.append(ego_depth)
        visuals = {
            'exo_rgb' : self.agentview_images[-1],  # 'exo rgb'
            'exo_rgb_boxed' : exo_image_boxed,  # 'exo rgb boxed'
            'exo_depth' : exo_depth,  # 'exo depth'
            'exo_seg' : exo_seg,  # 'exo seg'

            'ego_rgb' : self.eye_in_hand_images[-1],  # 'ego rgb'
            'ego_rgb_boxed' : ego_image_boxed,  # 'ego rgb boxed'
            'ego_depth' : ego_depth,  # 'ego depth'
            'ego_seg' : ego_seg,  # 'ego seg'
        }
        return visuals

    def process_segmentations(self, obs, seg_to_mask_fn, get_bbox_fn):
        ego_objs = {}
        exo_objs = {}
        ego_seg = np.zeros_like(obs['robot0_eye_in_hand_segmentation_robot0_eye_in_hand'])[:,:,0]
        exo_seg = np.zeros_like(obs['agentview_segmentation_agentview'])[:,:,0]

        for ind, noun in enumerate(self.nouns):
            label = self.remapper[noun]

            seg = get_visual(obs, 'robot0_eye_in_hand_segmentation_robot0_eye_in_hand')
            seg_mask = seg_to_mask_fn(seg, self.id_list, noun.replace(' ', '_'))
            ego_seg += seg_mask * ((ind + 1) * 10)
            ego_box = get_bbox_fn(seg_mask)

            seg = get_visual(obs, 'agentview_segmentation_agentview')
            seg_mask = seg_to_mask_fn(seg, self.id_list, noun.replace(' ', '_'))
            exo_seg += seg_mask * ((ind + 1) * 10)
            exo_box = get_bbox_fn(seg_mask)

            if ego_box is not None:
                ego_objs[label] = [(ind + 1) * 10, list(ego_box)]
            if exo_box is not None:
                exo_objs[label] = [(ind + 1) * 10, list(exo_box)]

        return ego_objs, exo_objs, ego_seg.astype(np.uint8), exo_seg.astype(np.uint8)

    def monitor_gripper_collision(self, env):
        # Access the simulation data
        sim = env.sim
        # Iterate through all contacts
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            
            # Get the geom names involved in the contact
            geom1 = sim.model.geom_id2name(contact.geom1)
            geom2 = sim.model.geom_id2name(contact.geom2)
            
            # Check if the contact is between the gripper and the object
            if ("gripper0_finger" in geom1) and ("gripper" not in geom2):
                col_object = self.get_collision_object(contact.geom2)
                if col_object in self.subgoal_object_counter:
                    self.subgoal_object_counter[col_object] += 1

            elif ("gripper0_finger" in geom2) and ("gripper" not in geom1):
                col_object = self.get_collision_object(contact.geom1)
                if col_object in self.subgoal_object_counter:
                    self.subgoal_object_counter[col_object] += 1

    def get_collision_object(self, contact_geom):
        for key, values in self.id_list.items():
            if contact_geom in values:
                label = self.remapper[key]
                return label
        return None

    def update_subgoal_state(self, satisfied_subgoals):
        if len(satisfied_subgoals) > 0 and satisfied_subgoals[-1] != self.cur_subgoal:
            self.cur_subgoal = satisfied_subgoals[-1]
            # this makes for a new subgoal on an object
            self.oc_satisfied_subgoals['robot'] = self.cur_subgoal

            # update all relevant objects and reset the sbugoal object_counter
            for noun in self.nouns:
                label = self.remapper[noun]
                if self.subgoal_object_counter[label] > self.subgoal_object_relevant_THRESHOLD:
                    self.oc_satisfied_subgoals[label] = self.cur_subgoal
                self.subgoal_object_counter[label] = 0
        return

    def update_collision_state_with_objs(self, visuals=None):
        for _, noun in enumerate(self.nouns):
            label = self.remapper[noun]
            if label in self.agentview_boxes[-1]:
                self.agentview_boxes[-1][label].append(self.oc_satisfied_subgoals[label])
            if label in self.eye_in_hand_boxes[-1]:
                self.eye_in_hand_boxes[-1][label].append(self.oc_satisfied_subgoals[label])
        
        if visuals is not None:
            exo_image_contact = visuals['exo_rgb'].copy()
            ego_image_contact = visuals['ego_rgb'].copy()
            for label, data in self.agentview_boxes[-1].items():
                ind, bbox, contact = data
                color = get_color4label(label)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

                x_min, y_min, w, h = bbox
                x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
                x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
                cv2.rectangle(exo_image_contact, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                cv2.putText(exo_image_contact, str(contact), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

            for label, data in self.eye_in_hand_boxes[-1].items():
                ind, bbox, contact = data
                color = get_color4label(label)
                color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 

                x_min, y_min, w, h = bbox
                x_max, y_max = int((x_min + w)*IMAGE_RESOLUTION), int((y_min + h)*IMAGE_RESOLUTION)
                x_min, y_min = int(x_min*IMAGE_RESOLUTION), int(y_min*IMAGE_RESOLUTION)
                cv2.rectangle(ego_image_contact, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                cv2.putText(ego_image_contact, str(contact), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

            visuals['exo_rgb_boxed_contact'] = exo_image_contact  # 'exo rgb boxed'
            visuals['ego_rgb_boxed_contact'] = ego_image_contact  # 'ego rgb boxed'
        return

    def get_data(self):
        """Return all the recorded data (optional to implement)."""
        data = {
            'states': self.states,
            'actions': self.actions,
            'ee_states': self.ee_states,
            'gripper_states': self.gripper_states,
            'joint_states': self.joint_states,
            'robot_states': self.robot_states,
            'agentview_images': self.agentview_images,
            'eye_in_hand_images': self.eye_in_hand_images,
            'agentview_depths': self.agentview_depths,
            'eye_in_hand_depths': self.eye_in_hand_depths,
            'agentview_segs': self.agentview_segs,
            'eye_in_hand_segs': self.eye_in_hand_segs,
            'agentview_boxes': self.agentview_boxes,
            'eye_in_hand_boxes': self.eye_in_hand_boxes,
        }
        return data


import matplotlib.cm as cm
# deterministic shuffling of values to map each geom ID to a random int in [0, 255]
rstate = np.random.RandomState(seed=8)
inds = np.arange(256)
rstate.shuffle(inds)

def segmentation_to_rgb(seg_im):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    # seg_im = seg_im[:,:,0]
    seg_im = np.mod(seg_im, 256)
    # use @inds to map each geom ID to a color
    mapped = (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]
    return mapped

def imshow(name, image):
    # image = cv2.rotate(image.astype(np.uint8), cv2.ROTATE_180)
    cv2.imshow(name, image)

import spacy
# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

def process_nonessential(noun):
    to_removes = ['both', 'the', 'a', 'an']
    # Split the string into words
    words = noun.split()
    # Filter out the words that need to be removed
    filtered_words = [word for word in words if word not in to_removes]
    # Join the filtered words back into a string
    return ' '.join(filtered_words)

mapper = {
    'cream cheese box': 'cream_cheese',
    'moka pot': 'moka_pot',
    'moka pots': 'moka_pot',
    'white mug': 'porcelain_mug',
    'yellow and white mug': 'white_yellow_mug',
    'right plate': 'plate_1',
    'left plate': 'plate_2 ',
    'cookie box': 'cookies',
    'middle black bowl': 'akita_black_bowl_2',
    'back black bowl': 'akita_black_bowl_3',
    'frying pan': 'chefmate_8_frypan',
    'right moka pot': 'moka_pot_1',
    'cabinet shelf': 'wooden_two_layer_shelf',
    'left bowl': 'akita_black_bowl_1',
    'right bowl': 'akita_black_bowl_2',
    'red mug': 'red_coffee_mug'
}

SIMPLIFYING_DICT = {
    'akita black bowl': 'bowl',
    'wooden cabinet': 'cabinet',
    'wine bottle': 'bottle',
    'wine rack': 'rack',
    'white cabinet': 'cabinet'
}

def check_contact(env, id_list):
    # Access the simulation data
    sim = env.sim
    touched = False
    # Iterate through all contacts
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        
        # Get the geom names involved in the contact
        geom1 = sim.model.geom_id2name(contact.geom1)
        geom2 = sim.model.geom_id2name(contact.geom2)
        
        # Check if the contact is between the gripper and the object
        if ("gripper0_finger" in geom1) and ("gripper" not in geom2):
            # print(geom1, geom2)
            # print(f"Object '{geom2}' is in contact with the gripper!")
            # Extract the contact position (in world coordinates)
            # contact_position = contact.pos  # This gives the x, y, z position of contact in world space
            # print(f"Contact position: {contact_position}")
            # Create a temporary site (red sphere) at the contact point
            # site_id = sim.model.site_name2id("gripper0_grip_site")
            # sim.model.site_rgba[site_id] = np.array([1, 0, 0, 1])  # Red color
            # sim.data.site_xpos[site_id] = contact_position
            # sim.forward()

            for key, values in id_list.items():
                if contact.geom2 in values:
                    return key
            
            return None # table or background

        elif ("gripper0_finger" in geom2) and ("gripper" not in geom1):
            # print(geom1, geom2)
            # print(f"Object '{geom1}' is in contact with the gripper!")
            # Extract the contact position (in world coordinates)
            # contact_position = contact.pos  # This gives the x, y, z position of contact in world space
            # print(f"Contact position: {contact_position}")
            # Create a temporary site (red sphere) at the contact point
            # site_id = sim.model.site_name2id("gripper0_grip_site")
            # sim.model.site_rgba[site_id] = np.array([1, 0, 0, 1])  # Red color
            # sim.data.site_xpos[site_id] = contact_position

            # sim.forward()

            for key, values in id_list.items():
                if contact.geom1 in values:
                    return key

            return None # table or background

    # print("Gripper is not in contact with anything.")
    return None

def extract_clean_nouns(sentence):
    doc = nlp(sentence)
    # Extract noun phrases and standalone nouns
    nouns = [chunk.text for chunk in doc.noun_chunks]
    # Remove determiners like 'a', 'an', 'the'
    cleaned_nouns = [' '.join(token.text for token in nlp(noun) if token.pos_ != 'DET') for noun in nouns]
    cleaned_nouns = [process_nonessential(noun) for noun in nouns]
    return cleaned_nouns

# Define the JSON file path
json_file = "nouns_storage.json"
# Function to save nouns to JSON
def save_nouns_to_json(noun_dict, sentence, nouns):
    # Add the new sentence and its nouns
    noun_dict[sentence] = nouns

    # Write back to the JSON file
    with open(json_file, "w") as file:
        json.dump(noun_dict, file, indent=4)

# Function to retrieve nouns from JSON
def get_nouns_from_json(json_file):
    if os.path.exists(json_file):
        with open(json_file, "r") as file:
            data = json.load(file)
        return data
    else:
        return {}


def seg_to_mask(seg_image, id_list, goal):
    ids = id_list[goal]
    # Create a mask
    mask = np.isin(seg_image, ids)
    return mask


def get_bbox(mask, height=256, width=256):
    """
    Extract bounding boxes and the overarching bounding box from a binary mask.

    Args:
        mask (np.ndarray): Binary mask as a NumPy array (2D).
                          Non-zero pixels are considered part of the mask.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
            - A list of bounding boxes in (x_min, y_min, x_max, y_max) format.
            - The overarching bounding box in (x_min, y_min, x_max, y_max) format.
    """
    H, W = height, width

    # Ensure the mask is binary
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract individual bounding boxes
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 10:
            bounding_boxes.append((x, y, x + w, y + h))

    # Compute the overarching bounding box
    if bounding_boxes:
        x_min = min(bbox[0] for bbox in bounding_boxes)
        y_min = min(bbox[1] for bbox in bounding_boxes)
        x_max = max(bbox[2] for bbox in bounding_boxes)
        y_max = max(bbox[3] for bbox in bounding_boxes)
        x_min, y_min, x_max, y_max = x_min/W, y_min/H, x_max/W, y_max/H
        box = (x_min, y_min, x_max - x_min, y_max - y_min)
        overarching_box = np.array(box)
    else:
        # If no bounding boxes were found
        overarching_box = None

    return overarching_box

DEBUG_MODE = False
from mem_task_object_nouns import TASK_NOUN_DICT

import torch
import imageio
import re

def set_init_state(env, init_state):
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    env._check_success()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]

def remove_initial_noops(actions, threshold=1e-4):
    """
    Removes leading no-op actions (based on norm of action[:-1]).

    Args:
        actions (list or np.ndarray): List/array of actions.
        threshold (float): Norm threshold to consider as a no-op.

    Returns:
        np.ndarray: Filtered actions after removing initial no-ops.
    """
    actions = np.array(actions)
    start_idx = 0

    for i, action in enumerate(actions):
        if np.linalg.norm(action[:-1]) >= threshold:
            start_idx = i
            break

    return actions[start_idx:], start_idx

TASK_LENGTHS = {
    'pick up the bowl and place it back on the plate': 180,
    'lift the bottle and put it down on the plate': 285,
    'lift the bowl and place it back on the plate 3 times': 320,
    'pick up the bottle and put it down the plate 3 times': 365,
    'lift the bowl and place it back on the plate 5 times': 445,
    'pick up the bowl and place it on the plate 7 times': 650,
    'swap the 2 bowls on their plates using the empty plate': 475,
    'rotate the 3 bowls on their plates from left to right using the empty plate': 670,
    'put the cream cheese in the nearest basket and place that basket in the center': 365,
    'put the cream cheese in the nearest basket and place the empty basket in the center': 390,
}

from collections import defaultdict
import pickle

# Convert defaultdicts to regular dicts for saving
def recursive_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: recursive_to_dict(v) for k, v in d.items()}
    return d


def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # # Create target directory
    os.makedirs(args.libero_result_dir, exist_ok=True)
    
    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    
    history = {}
    all_trajs = {}

    # Iterate through the task suites
    for task_id in tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        task_description = task_description.strip()
        goal_length = env.get_goal_sequence_len()

        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]
        os.makedirs(os.path.join(args.libero_result_dir, task.name), exist_ok=True)
    
        # Print the trajectories overall details of the task
        task_success = 0
        tiered_task_success = 0
        # Nested defaultdict that automatically creates the inner dict structure
        all_trajs = defaultdict(lambda: defaultdict(dict))
        print(f"Task: {task_description}")
        print(f"There are {min(args.eval_runs, len(orig_data.keys()))} episodes to run.")
        key_indices = orig_data.keys()
        key_indices = sorted(key_indices, key=natural_sort_key)

        test_indices = []
        if args.eval_mode == 'seen':
            test_indices = np.arange(0, min(args.eval_runs, len(orig_data.keys())))
        elif args.eval_mode == 'unseen':
            test_indices = np.arange(min(args.eval_runs, 20), 0, -1)
            test_indices = len(orig_data.keys()) - test_indices
    
        for idx in tqdm(test_indices): 
            # Get demo data
            # demo_data = orig_data[f"demo_{demo_seq_idx+1}"]
            if "init_state" in orig_data[key_indices[idx]]:
                init_state = orig_data[key_indices[idx]]["init_state"][()]
            else:
                init_state = orig_data[key_indices[idx]]["states"][()][0]
            
            orig_actions, skipped_num = remove_initial_noops(orig_data[key_indices[idx]]["actions"][()])
            orig_actions = orig_actions.tolist()
            action = [0,0,0,0,0,0,0]
            for i in range(20):
                orig_actions.append(action)

            recorder = TrajectoryRecorder(
                    image_resolution=IMAGE_RESOLUTION
            )
            recorder.reset_object_mappers(env)
            
            done = False
            retry = 0
            frames = []
            while True:

                prev_contact_key = None
                curr_contact_key = None
                prev_grip = False
                curr_grip = False
                early_terminated = -1

                # Reset environment, set initial state, and wait a few steps for environment to settle
                env.reset()
                set_init_state(env, init_state)
                env.reset_subgoal_progress()
                
                # print(get_task_init_states()[0])
                for _ in range(20):
                    obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

                robot_trajs = []

                recorder.reset_trackstate()
                for action in tqdm(orig_actions):

                    # Record the state information
                    visuals = recorder.record(obs, action, seg_to_mask_fn=seg_to_mask, get_bbox_fn=get_bbox)
                    robot_trajs.append([obs['robot0_eef_pos'], action])

                    # # Get the collision check
                    # contact_geom, geom = collision_check(env)
                    # curr_contact_key = recorder.get_collision_object(contact_geom)

                    # monitor object collision
                    recorder.monitor_gripper_collision(env)
                    """
                        extended interaction with an object right before a newly satisfied subgoal -> completed object-centric subgoal
                    """
                    env._check_success(inc=True)
                    recorder.update_subgoal_state(env.get_satisfied_subgoals(task_description))
                    recorder.update_collision_state_with_objs(visuals)

                    if DEBUG_MODE:
                        # Get the corresponding visualizations
                        row1 = [
                            visuals['exo_rgb'],  # 'exo rgb'
                            visuals['exo_rgb_boxed'],  # 'exo rgb'
                            visuals['exo_rgb_boxed_contact'],  # 'exo rgb'
                            visuals['exo_depth'],  # 'exo depth'
                            visuals['exo_seg'],  # 'exo seg'
                        ]

                        row2 = [
                            visuals['ego_rgb'],  # 'ego rgb'
                            visuals['ego_rgb_boxed'],  # 'ego rgb'
                            visuals['ego_rgb_boxed_contact'],  # 'ego rgb'
                            visuals['ego_depth'],  # 'ego depth'
                            visuals['ego_seg'],  # 'ego seg'
                        ]

                        # Function to preprocess images (convert grayscale to RGB and resize)
                        def preprocess_image(image, target_size=(300, 300)):
                            if len(image.shape) == 2:  # Convert grayscale to RGB
                                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, target_size)
                            return image

                        # Process all images to uniform size and color format and concatenate horizontally
                        row1_concat = np.hstack([preprocess_image(img) for img in row1])
                        row2_concat = np.hstack([preprocess_image(img) for img in row2])
                        print(row1_concat.shape)
                        # Concatenate vertically to form a 2x4 grid
                        final_image = np.vstack([row1_concat, row2_concat])
                        imshow(task_description, final_image)
                        cv2.waitKey(1)
                        frames.append(visuals['exo_rgb'])

                    else:
                        row1 = [
                            visuals['exo_rgb'],  # 'exo rgb'
                            visuals['exo_rgb_boxed'],  # 'exo rgb'
                            visuals['exo_rgb_boxed_contact'],  # 'exo rgb'
                            visuals['exo_depth'],  # 'exo depth'
                            visuals['exo_seg'],  # 'exo seg'
                        ]

                        row2 = [
                            visuals['ego_rgb'],  # 'ego rgb'
                            visuals['ego_rgb_boxed'],  # 'ego rgb'
                            visuals['ego_rgb_boxed_contact'],  # 'ego rgb'
                            visuals['ego_depth'],  # 'ego depth'
                            visuals['ego_seg'],  # 'ego seg'
                        ]

                        # Function to preprocess images (convert grayscale to RGB and resize)
                        def preprocess_image(image, target_size=(300, 300)):
                            if len(image.shape) == 2:  # Convert grayscale to RGB
                                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image = cv2.resize(image, target_size)
                            return image

                        # Process all images to uniform size and color format and concatenate horizontally
                        row1_concat = np.hstack([preprocess_image(img) for img in row1])
                        row2_concat = np.hstack([preprocess_image(img) for img in row2])
        
                        # Concatenate vertically to form a 2x4 grid
                        final_image = np.vstack([row1_concat, row2_concat])
                        # frames.append(final_image)

                    # Go to next state information
                    # try:
                    action[-1] = action[-1] if action[-1] != 0 else -1
                    obs, reward, done, info = env.step(action)
                    # except:
                    #     print(action)
                    #     break
                
                satisfied_subgoals = env.get_satisfied_subgoals(task_description)

                # Run when finished
                break

            cv2.destroyAllWindows()
            # Save successful trajectory
            if done:
                all_trajs[task_description][key_indices[idx]]['robot_traj'] = robot_trajs
                all_trajs[task_description][key_indices[idx]]['success'] = True
                all_trajs[task_description][key_indices[idx]]['tiered_success'] = 1
                all_trajs[task_description][key_indices[idx]]['satisfied_subgoals'] = satisfied_subgoals
                all_trajs[task_description][key_indices[idx]]['agentview_boxes'] = recorder.agentview_boxes

                print(env.get_satisfied_subgoals(task_description))
                print("Tiered_success =", len(satisfied_subgoals)/goal_length)

                task_success += 1
                tiered_task_success += 1
            else:
                all_trajs[task_description][key_indices[idx]]['robot_traj'] = robot_trajs
                all_trajs[task_description][key_indices[idx]]['success'] = False
                all_trajs[task_description][key_indices[idx]]['tiered_success'] = len(satisfied_subgoals) / goal_length
                all_trajs[task_description][key_indices[idx]]['satisfied_subgoals'] = satisfied_subgoals
                all_trajs[task_description][key_indices[idx]]['agentview_boxes'] = recorder.agentview_boxes

                print(env.get_satisfied_subgoals(task_description))
                print("Tiered_failed =", len(satisfied_subgoals)/goal_length)

                # No full success; compute tiered success based on subgoals
                tiered_task_success += len(satisfied_subgoals) / goal_length

            # Update progress history
            history[f'Task {task_id}: {task_description}'] = f'{tiered_task_success}/{idx + 1}'
            # Print current progress
            for name, result in history.items():
                print(f'{name}\t{result}')
    
    # Save
    save_filename = f"all_trajs_{args.eval_mode}.pkl"
    save_path = os.path.join(args.libero_result_dir, save_filename)
    with open(save_path, 'wb') as f:
        pickle.dump(recursive_to_dict(all_trajs), f)
        
    if DEBUG_MODE:
        with imageio.get_writer('output_video.mp4', fps=30) as writer:
            for frame in frames:
                writer.append_data(frame)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_mem", "libero_relation", "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_result_dir", type=str,
                        help="Path to trajectory videos. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    parser.add_argument('--eval_mode', type=str, choices=['seen', 'unseen'], help='Type of eval run')    
    parser.add_argument('--eval_runs', default=20, type=int, help='Maximum number of replays')    
    parser.add_argument('--debug_mode', action='store_true')    
    args = parser.parse_args()

    DEBUG_MODE = args.debug_mode
    # Start data regeneration
    main(args)

