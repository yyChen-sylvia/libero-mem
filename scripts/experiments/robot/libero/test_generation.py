"""
Regenerates a LIBERO dataset (HDF5 files) by replaying demonstrations in the environments.

Notes:
    - We save image observations at 256x256px resolution (instead of 128x128).
    - We filter out transitions with "no-op" (zero) actions that do not change the robot's state.
    - We filter out unsuccessful demonstrations.
    - In the LIBERO HDF5 data -> RLDS data conversion (not shown here), we rotate the images by
    180 degrees because we observe that the environments return images that are upside down
    on our platform.

Usage:
    python experiments/robot/libero/regenerate_libero_dataset.py \
        --libero_task_suite [ libero_spatial | libero_object | libero_goal | libero_10 ] \
        --libero_raw_data_dir <PATH TO RAW HDF5 DATASET DIR> \
        --libero_target_dir <PATH TO TARGET DIR>

    Example (LIBERO-Spatial):
        python experiments/robot/libero/regenerate_libero_dataset.py \
            --libero_task_suite libero_spatial \
            --libero_raw_data_dir ./LIBERO/libero/datasets/libero_spatial \
            --libero_target_dir ./LIBERO/libero/datasets/libero_spatial_no_noops

"""

from scipy.spatial.transform import Rotation as R
import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import robosuite.utils.transform_utils as T
import tqdm
from libero.libero import benchmark
from flow import TorchFlow
from torchvision.utils import flow_to_image

from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
)
import sys
sys.path.append('/home/nhatchung/Desktop/dataspace/libero/LIBERO')

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

import matplotlib.pyplot as plt

def draw_eef_data(eef_data_list, out_dir):
    """
    Draws and saves 7 separate plots (one for each dimension) 
    of end-effector data over time.

    Parameters
    ----------
    eef_data_list : list of lists
        A Python list (length T) of sub-lists, where each sub-list
        has 7 values, e.g. [x, y, z, r_x, r_y, r_z, gripper].
    """
    # Convert the list of lists to a NumPy array
    eef_data = np.array(eef_data_list, dtype=float)  # shape = (T, 7)

    # Define labels for each dimension
    dimension_names = ["x", "y", "z", "r_x", "r_y", "r_z", "gripper"]
    
    # Create a time axis (0, 1, 2, ... T-1)
    timesteps = np.arange(eef_data.shape[0])

    # Loop over each dimension and plot
    for i in range(7):
        # Create a new figure
        plt.figure(figsize=(6, 4))

        # Plot the data for this dimension
        plt.plot(timesteps, eef_data[:, i], label=dimension_names[i])
        
        # Labeling
        # plt.title(f"EEF {dimension_names[i]} over Time")
        # plt.xlabel("Timestep")
        # plt.ylabel(dimension_names[i])
        plt.xlim(0, len(eef_data)) 
        plt.legend()
        plt.grid()

        # Save each figure as a separate file
        filename = os.path.join(out_dir, f"eef_{dimension_names[i]}.png")
        plt.savefig(filename)
        
        # Close the figure to free memory
        plt.close()

    print("Saved 7 plots (one per EEF dimension) to separate files.")

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
def map_to_base(noun):
    if noun in mapper:
        return mapper[noun]
    return noun

REVERSE_SCENARIOS = ['put the red mug on the left plate',
                     'put the red mug on the right plate',
                     'put the white mug on the left plate',
                     'put the white mug on the right plate',
                     'put the yellow and white mug on the right plate', 
                     'pick up the book in the middle and place it on the cabinet shelf',
                     'pick up the book on the left and place it on top of the shelf',
                     'pick up the book on the right and place it on top of the shelf']

SIMPLIFYING_DICT = {
    'akita black bowl': 'bowl',
    'wooden cabinet': 'cabinet',
    'wine bottle': 'bottle',
    'wine rack': 'rack'
}

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

def process_to_base(nouns, task=''):
    remapper = {}
    cleaned_nouns = []
    for noun in nouns:
        if noun == 'left plate':
            if task in REVERSE_SCENARIOS:
                clean_noun = 'plate_1'
            else:
                clean_noun = map_to_base(noun)
        elif noun == 'right plate':
            if task in REVERSE_SCENARIOS:
                clean_noun = 'plate_2'
            else:
                clean_noun = map_to_base(noun)
        # elif noun == 'book':
        #     if task in REVERSE_SCENARIOS:
        #         clean_noun = 'book'
        #     else:
        #         clean_noun = map_to_base(noun)
        else:
            clean_noun = map_to_base(noun)
        remapper[clean_noun] = noun
        cleaned_nouns.append(clean_noun)
    return cleaned_nouns, remapper

color4label = {}
def get_color4label(label):
    if label in color4label:
        return color4label[label]
    else:
        new_color = np.random.randint(0, 255, size=(3, ))
        color4label[label] = tuple(new_color.astype(np.int32))
        return color4label[label]

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

DEBUG_MODE = False
task_noun_dict = {
    # libero goal
    'open the middle drawer of the cabinet': ['robot', 'middle drawer', 'cabinet 1'],
    'put the bowl on the stove': ['robot', 'bowl 1', 'stove'],
    'put the wine bottle on top of the cabinet': ['robot', 'bottle 1', 'cabinet 1'],
    'open the top drawer and put the bowl inside': ['robot', 'top drawer', 'bowl 1'],
    'put the bowl on top of the cabinet': ['robot', 'bowl 1', 'cabinet 1'],
    'push the plate to the front of the stove': ['robot', 'plate 1', 'stove'],
    'put the cream cheese in the bowl': ['robot', 'cream cheese 1', 'bowl 1'],
    'turn on the stove': ['robot', 'stove'],
    'put the bowl on the plate': ['robot', 'bowl 1', 'plate 1'],
    'put the wine bottle on the rack': ['robot', 'bottle 1', 'rack 1'],

    # libero relation
    'place the black bowl behind the basket onto the left plate': ['robot', 'bowl 2', 'plate 1'],
    'place the black bowl in front of the basket onto the right plate': ['robot', 'bowl 1', 'plate 2'],
    'place the cream cheese box behind the moka pot onto the right bowl': ['robot', 'cream cheese 1', 'bowl 2'],
    'place the cream cheese box in front of the moka pot onto the left bowl': ['robot', 'cream cheese 2', 'bowl 1'],
    'place the left bowl on the right bowl': ['robot', 'bowl 1', 'bowl 2'],
    'place the bottle behind the cabinet onto the right plate': ['robot', 'bottle 1', 'cabinet 1', 'plate 2'],
    'place the bottle behind the cabinet to the left of the basket': ['robot', 'bottle 1', 'cabinet 1', 'basket 1'],
    'place the bottle behind the cabinet to the front of the basket': ['robot', 'bottle 1', 'cabinet 1', 'basket 1'],
    'put the bottle from the basket onto the cabinet': ['robot', 'bottle 1', 'cabinet 1', 'basket 1'],
    'place the cream cheese and the bowl into the basket': ['robot', 'cream cheese 1', 'bowl 1', 'basket 1'],
    'place the cream cheese behind the cabinet to the left of the basket': ['robot', 'cream cheese 1', 'basket 1'],
    'place the cream cheese behind the cabinet to the right of the basket': ['robot', 'cream cheese 1', 'basket 1'],
    'place the cream cheese behind the cabinet to the front of the basket': ['robot', 'cream cheese 1', 'basket 1'],
    'place the cream cheese behind the cabinet to behind the basket': ['robot', 'cream cheese 1', 'basket 1'],
    'place the bowl on the plate behind the cabinet': ['robot', 'bowl 1', 'plate 1', 'cabinet 1'],
    'stack the left bowl on the right bowl behind the cabinet': ['robot', 'bowl 1', 'bowl 2', 'cabinet 1'],
    'place the bottle onto the cabinet and put the left bowl behind the cabinet to the basket': ['robot', 'bottle 1', 'bowl 1', 'cabinet 1'],
    'put the left bowl to behind the cabinet and put the right bowl to behind the cabinet': ['robot', 'bowl 1', 'bowl 2', 'cabinet 1'],
    'stack the left bowl onto right bowl and put them behind the basket': ['robot', 'bowl 1', 'bowl 2', 'basket 1'],
    'stack the right cream cheese onto the left cream cheese and place them behind the cabinet': ['robot', 'cream cheese 1', 'cream cheese 2', 'cabinet 1'],
}

import imageio

def get_task_nouns(task_description):
    if task_description in task_noun_dict:
        return task_noun_dict[task_description]
    print(task_description)
    1/0
    return []

# Generate a colormap with 256 unique colors
np.random.seed(1)  # Ensure reproducibility
colormap = np.random.randint(0, 256, size=(256, 3), dtype=np.uint8)  # 256x3 (R, G, B)

def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")

    # # Create target directory
    if not DEBUG_MODE:
        if os.path.isdir(args.libero_target_dir):
            user_input = input(f"Target directory already exists at path: {args.libero_target_dir}\nEnter 'y' to overwrite the directory, or anything else to exit: ")
            if user_input != 'y':
                exit()
        os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = f"./{args.libero_task_suite}_metainfo.json"
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    
    # Setup
    num_replays = 0
    num_success = 0
    num_noops = 0

    # Sentence2Nouns
    noun_dict = get_nouns_from_json(json_file)

    # # # Get Flow model
    # flow_model = TorchFlow()
    frames = []
    counter = 0
    noop_checker = NoOpChecker()
    history = {}
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # counter += 1
        # if counter > 1:
        #     break
        # Get task in suite
        task = task_suite.get_task(task_id)
        env, task_description = get_libero_env(task, "llava", resolution=IMAGE_RESOLUTION)
        task_description = task_description.strip()
        print('Running', task_description)
        # if task_description != 'put the bowl on the stove':
        #     continue
        
        # Get dataset for task
        orig_data_path = os.path.join(args.libero_raw_data_dir, f"{task.name}_demo.hdf5")
        assert os.path.exists(orig_data_path), f"Cannot find raw data file {orig_data_path}."
        orig_data_file = h5py.File(orig_data_path, "r")
        orig_data = orig_data_file["data"]

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}")
        os.makedirs(new_data_path, exist_ok=True)

        # # Print all mappings
        # if task_description not in noun_dict:
        if True:
            if 'relation' in args.libero_task_suite:
                nouns = ['gripper0']
            else:
                nouns = ['top_drawer', 'middle_drawer', 'bottom_drawer', 'flat_stove', 'gripper0']
            remapper = {}
            remapper['gripper0'] = 'robot'
            remapper['top_drawer'] = 'top drawer'
            remapper['middle_drawer'] = 'middle drawer'
            remapper['bottom_drawer'] = 'bottom drawer'
            remapper['flat_stove'] = 'stove'

            task_nouns = get_task_nouns(task_description)
        
        # Example usage
        # save_nouns_to_json(noun_dict, task_description, nouns)
        # print(nouns)

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
        # print(geom_ids)
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
                    id_list[re_noun].append(geom_name2id[geom])
        nouns+= remain_obj_keys
        remapper = revise_remapper(remapper)
        if DEBUG_MODE:
            print(nouns)
            print(remapper)

        # almost had to run to another place
        task_success, task_replays = 0, 0
        for i in range(len(orig_data.keys())):
            if DEBUG_MODE:
                if i >= 1:
                    break
            # Get demo data
            if "demo_0" not in list(orig_data.keys()):
                demo_data = orig_data[f"demo_{i+1}"]
            else:
                demo_data = orig_data[f"demo_{i}"]
            orig_actions = demo_data["actions"][()]
            orig_actions = noop_checker.filter_noops(orig_actions)
            orig_states = demo_data["states"][()]

            # Reset environment, set initial state, and wait a few steps for environment to settle
            env.reset()
            env.set_init_state(orig_states[0])
            for _ in range(10):
                obs, reward, done, info = env.step(get_libero_dummy_action("llava"))

            # Set up new data lists
            states = []
            actions = []
            ee_states = []
            gripper_states = []
            joint_states = []
            robot_states = []
            agentview_images = []
            eye_in_hand_images = []
            agentview_depths = []
            eye_in_hand_depths = []
            agentview_segs = []
            eye_in_hand_segs = []
            agentview_boxes = []
            eye_in_hand_boxes = []
            agentview_flows = []
            eye_in_hand_flows = []
            cur_obs = None
            eef_data = []
            # Replay original demo actions in environment and record observations
            for _ind, action in enumerate(orig_actions):
                # Skip transitions with no-op actions
                # prev_action = actions[-1] if len(actions) > 0 else None
                # if noop_checker.is_noop(action, prev_action):
                #     print(f"\tSkipping no-op action: {action}")
                #     num_noops += 1
                #     continue
                # print(f'\tNot skippeed: {action}')

                # if cur_obs is not None:
                #     if action[-1] > 0:
                #         print(obs["robot0_gripper_qpos"][0]-cur_obs["robot0_gripper_qpos"][0], np.abs(obs["robot0_gripper_qpos"][0]-cur_obs["robot0_gripper_qpos"][0]) < 0.00002)

                if states == []:
                    # In the first timestep, since we're using the original initial state to initialize the environment,
                    # copy the initial state (first state in episode) over from the original HDF5 to the new one
                    states.append(orig_states[0])
                    if "robot_states" in demo_data:
                        robot_states.append(demo_data["robot_states"][0])
                    else:
                        robot_states.append(
                            np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                        )
                else:
                    # For all other timesteps, get state from environment and record it
                    states.append(env.sim.get_state().flatten())
                    robot_states.append(
                        np.concatenate([obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]])
                    )

                # Record original action (from demo)
                actions.append(action)

                # Record data returned by environment
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])
                joint_states.append(obs["robot0_joint_pos"])
                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )

                # Execute demo action in environment
                cur_obs = obs
                # --- EEF Position ---
                x, y, z = obs["robot0_eef_pos"]

                # --- EEF Orientation (Quaternion) -> Euler ---
                w, qx, qy, qz = obs["robot0_eef_quat"]
                rotation = R.from_quat([qx, qy, qz, w])  # reorder to x,y,z,w
                r_x, r_y, r_z = rotation.as_euler('xyz', degrees=False)

                # --- Gripper States ---
                gripper_qpos = action[-1]

                print("EEF Position (XYZ):", x, y, z)
                print("EEF Orientation (rX, rY, rZ) in radians:", r_x, r_y, r_z)
                print("Gripper joint positions:", gripper_qpos)
                eef_point = [x,y,z,r_x,r_y,r_z,gripper_qpos]
                eef_data.append(eef_point)
                print("----------")
                
                # print(action.tolist())
                nxt_obs, reward, done, info = env.step(action.tolist())
                # Execute a couple of NOOPS inbetween
                # for _ in range(1):
                #     _action = action.copy()
                #     _action[:-1] = 0
                #     nxt_obs, reward, done, info = env.step(_action.tolist())
                obs = nxt_obs

                agentview_images.append(cv2.rotate(cur_obs["agentview_image"][:,::-1], cv2.ROTATE_180))
                eye_in_hand_images.append(cv2.rotate(cur_obs["robot0_eye_in_hand_image"][:,::-1], cv2.ROTATE_180))

                # flow = flow_model.run(cur_obs["agentview_image"], obs["agentview_image"])
                # flow = flow[-1]
                # flow_img = flow_to_image(flow)[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                # flow_img = cv2.resize(flow_img, (IMAGE_RESOLUTION, IMAGE_RESOLUTION))
                # flow_img = cv2.rotate(flow_img, cv2.ROTATE_180)
                # agentview_flows.append(flow_img)
                # if DEBUG_MODE:
                #     imshow('exo flow', flow_img)

                # flow = flow_model.run(cur_obs["robot0_eye_in_hand_image"], obs["robot0_eye_in_hand_image"])
                # flow = flow[-1]
                # flow_img = flow_to_image(flow)[0].cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                # flow_img = cv2.resize(flow_img, (IMAGE_RESOLUTION, IMAGE_RESOLUTION))
                # flow_img = cv2.rotate(flow_img, cv2.ROTATE_180)
                # eye_in_hand_flows.append(flow_img)
                # if DEBUG_MODE:
                #     imshow('ego flow', flow_img)

                # imshow('robot0_eye_in_hand_segmentation_agentview', 
                #            segmentation_to_rgb(obs['robot0_eye_in_hand_segmentation_agentview']))
                # imshow('robot0_eye_in_hand_segmentation_robot0_eye_in_hand', 
                #            segmentation_to_rgb(obs['robot0_eye_in_hand_segmentation_robot0_eye_in_hand']))
                # imshow('agentview_segmentation_agentview', 
                #            segmentation_to_rgb(obs['agentview_segmentation_agentview']))

                ego_objs = {}
                exo_objs = {}
                ego_seg = np.zeros_like(cur_obs['robot0_eye_in_hand_segmentation_robot0_eye_in_hand'])[:,:,0]
                exo_seg = np.zeros_like(cur_obs['robot0_eye_in_hand_segmentation_robot0_eye_in_hand'])[:,:,0]
                for ind, noun in enumerate(nouns):
                    label = remapper[noun]
                    # if label not in task_nouns:
                    #     continue
                    seg = cur_obs['robot0_eye_in_hand_segmentation_robot0_eye_in_hand'][:,::-1]
                    seg = cv2.rotate(seg.astype(np.uint8), cv2.ROTATE_180)
                    seg_mask = seg_to_mask(seg, id_list, noun.replace(' ', '_'))
                    ego_seg = ego_seg + seg_mask*((ind+1)*10)
                    ego_box = get_bbox(seg_mask)
                    # imshow('ego ' + label, segmentation_to_rgb(seg_mask))
                    seg = cur_obs['agentview_segmentation_agentview'][:,::-1]
                    seg = cv2.rotate(seg.astype(np.uint8), cv2.ROTATE_180)
                    seg_mask = seg_to_mask(seg, id_list, noun.replace(' ', '_'))
                    exo_seg = exo_seg + seg_mask*((ind+1)*10)
                    exo_box = get_bbox(seg_mask)
                    # imshow('exo ' + label, segmentation_to_rgb(seg_mask))
                    if ego_box is not None:
                        ego_objs[label] = [(ind+1)*10, list(ego_box)]
                    if exo_box is not None:
                        exo_objs[label] = [(ind+1)*10, list(exo_box)]
                eye_in_hand_boxes.append(ego_objs)
                agentview_boxes.append(exo_objs)

                ego_seg, exo_seg = colormap[ego_seg.astype(np.uint8)], colormap[exo_seg.astype(np.uint8)]
                agentview_segs.append(exo_seg)
                eye_in_hand_segs.append(ego_seg)

                ego_image = cur_obs["robot0_eye_in_hand_image"][:,::-1,:]
                ego_image = cv2.rotate(ego_image.astype(np.uint8), cv2.ROTATE_180)
                relevant_ego_image = ego_image.copy()
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
                    
                    if label not in task_nouns:
                        continue
                    relevant_ego_image = relevant_ego_image
                    cv2.rectangle(relevant_ego_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                    text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                    cv2.putText(relevant_ego_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)


                exo_image = cur_obs["agentview_image"][:,::-1,:]
                exo_image = cv2.rotate(exo_image.astype(np.uint8), cv2.ROTATE_180)
                relevant_exo_image = exo_image.copy()
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

                    if label not in task_nouns:
                        continue
                    relevant_exo_image = relevant_exo_image
                    cv2.rectangle(relevant_exo_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)
                    text_position = (x_min, y_min - 10 if y_min - 10 > 10 else y_min + 10)
                    cv2.putText(relevant_exo_image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=color, thickness=2)

                exo_depth = cv2.rotate(min_max_normalize(cur_obs["agentview_depth"][:,::-1], old_min=0.97, old_max=1.0), cv2.ROTATE_180)
                ego_depth = cv2.rotate(min_max_normalize(cur_obs["robot0_eye_in_hand_depth"][:,::-1], old_min=0.71, old_max=1.0), cv2.ROTATE_180)
                agentview_depths.append(exo_depth)
                eye_in_hand_depths.append(ego_depth)

                if DEBUG_MODE:
                    # Collect images
                    row1 = [
                        agentview_images[-1],  # 'exo rgb'
                        exo_image,  # 'ego rgb boxed'
                        relevant_exo_image,
                        exo_depth,  # 'exo depth'
                        exo_seg,  # 'exo seg'
                    ]

                    row2 = [
                        eye_in_hand_images[-1] ,  # 'ego rgb'
                        ego_image,  # 'ego rgb boxed'
                        relevant_ego_image,
                        ego_depth,  # 'exo depth'
                        ego_seg,  # 'ego seg'
                    ]

                    # Function to preprocess images (convert grayscale to RGB and resize)
                    def preprocess_image(image, target_size=(300, 300)):
                        if len(image.shape) == 2:  # Convert grayscale to RGB
                            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                        image = cv2.resize(image, target_size)
                        return image

                    # Process all images to uniform size and color format and concatenate horizontally
                    row1_concat = np.hstack([preprocess_image(img) for img in row1])
                    row2_concat = np.hstack([preprocess_image(img) for img in row2])
    
                    # Concatenate vertically to form a 2x4 grid
                    final_image = np.vstack([row1_concat, row2_concat])
                    final_image = cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
                    imshow('all', final_image)
                    cv2.imwrite(os.path.join(new_data_path, str(_ind).zfill(4) + '.jpg'), final_image)
                    frames.append(final_image)

                if DEBUG_MODE:
                    _key_ = cv2.waitKey(1)
                    if _key_ == ord('q'):
                        done = True
                        break
                    elif _key_ == ord('p'):
                        while cv2.waitKey(1) != ord('p'):
                            continue

            # Report total number of no-op actions filtered out so far
            print(f"  Total # no-op actions filtered out: {num_noops}")
            cv2.destroyAllWindows()
            draw_eef_data(eef_data, out_dir=new_data_path)

            if task_success >= 50:
                break

        # Close HDF5 files
        orig_data_file.close()
        print(f"Saved regenerated demos for task '{task_description}' at: {new_data_path}")



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_relation", "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    parser.add_argument('--debug_mode', action='store_true')    
    args = parser.parse_args()

    DEBUG_MODE = args.debug_mode
    # Start data regeneration
    main(args)
