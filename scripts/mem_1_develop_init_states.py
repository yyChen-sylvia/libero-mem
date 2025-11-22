"""
Generate and store initial states for training data and evaluation setups.

Usage examples:
    python mem_1_develop_init_states.py \
        --bddl-file "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.bddl" \
        --target-path "init-data"

    python mem_1_develop_init_states.py \
        --bddl-file "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate.bddl" \
        --target-path "init-data"

    python mem_1_develop_init_states.py \
        --bddl-file "KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate.bddl" \
        --target-path "init-data"

    python mem_1_develop_init_states.py \
        --bddl-file "KITCHEN_SCENE1_9_put_the_bowl_in_the_nearest_basket_and_place_that_basket_in_the_center.bddl" \
        --target-path "init-data"
"""


import argparse
import cv2
import datetime
import h5py
import init_path
import json
import numpy as np
import os
import robosuite as suite
import time
from glob import glob
from robosuite import load_controller_config
from robosuite.wrappers import DataCollectionWrapper, VisualizationWrapper
from robosuite.utils.input_utils import input2action


import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *

import imageio
def save_rollout_video(rollout_images, bddl_file):
    if bddl_file is not None:
        bddl_file = os.path.basename(bddl_file)[:-5]

    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./saved_video/"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{bddl_file}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def collect_init_states(
    env, arm, env_configuration, problem_info, remove_directory=[], bddl_file=None, video_saving=True
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    while True:
        no_collision = True
        env.reset()

        # Access MuJoCo sim
        sim = env.sim

        # Number of contacts (collisions) at this step
        print("Number of collisions:", sim.data.ncon)

        # Iterate through each contact
        table_collision = 0
        for i in range(sim.data.ncon):
            contact = sim.data.contact[i]
            geom1 = sim.model.geom_id2name(contact.geom1)
            geom2 = sim.model.geom_id2name(contact.geom2)
            print(f"Collision between: {geom1} and {geom2}")
            if 'plate' in geom1 and 'plate' in geom2:
                table_collision += 1
                break
            
        if table_collision == 0:
            break
            
    # reset_success = False
    # while not reset_success:
    #     try:
    #         env.reset()
    #         reset_success = True
    #     except:
    #         continue

    # ID = 2 always corresponds to agentview
    # env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal

    # Loop until we get a reset from the input or the task completes
    saving = True
    count = 0
    rollout_video = []
    print("Press 'q' to reset and 's' to save init state.")
    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Run environment step
        obs, reward, done, info = env.step([0,0,0,0,0,0,-1])
        # env.render()

        ego_view = obs['robot0_eye_in_hand_image'][:,::-1]
        exo_view = obs['agentview_image'][::-1,:]
        camera_view = cv2.hconcat([exo_view, ego_view])

        cv2.imshow('camera_view', cv2.cvtColor(camera_view, cv2.COLOR_RGB2BGR))
        _key_ = cv2.waitKey(1)
        if _key_ == ord('q'):
            saving = False
            break
        elif _key_ == ord('s'):
            saving = True
            break

    # cleanup for end of data collection episodes
    if not saving:
        remove_directory.append(env.ep_directory.split("/")[-1])
    env.close()
    return saving

import torch
def gather_inits(
    directory, out_dir, env_info, bddl_file, remove_directory=[]
):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    The strucure of the hdf5 file is as follows.

    data (group)
        date (attribute) - date of collection
        time (attribute) - time of collection
        repository_version (attribute) - repository version used during collection
        env (attribute) - environment name on which demos were collected

        demo1 (group) - every demonstration has a group
            model_file (attribute) - model xml string for demonstration
            states (dataset) - flattened mujoco states
            actions (dataset) - actions applied during demonstration

        demo2 (group)
        ...

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """

    file_name = os.path.basename(bddl_file)[:-5]
    all_inits = []
    for ep_directory in os.listdir(directory):
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        
        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            states.extend(dic["states"])

        all_inits.append(states[0])

    all_inits = np.array(all_inits)
    torch.save(all_inits, os.path.join(out_dir, file_name+'.pruned_init'))
    return

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target-path",
        type=str,
        default="Target path for saving inits",
    )
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="single-arm-opposed",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="agentview",
        help="Which camera to use for collecting demos",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="OSC_POSE",
        help="Choice of controller. Can be 'IK_POSE' or 'OSC_POSE'",
    )
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.5,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--num-demonstration",
        type=int,
        default=120,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument("--bddl-file", type=str)

    parser.add_argument("--vendor-id", type=int, default=9583)
    parser.add_argument("--product-id", type=int, default=50734)
    parser.add_argument("--video_saving", action='store_true')

    args = parser.parse_args()

    # Get controller config
    controller_config = load_controller_config(default_controller=args.controller)

    # Create argument configuration
    config = {
        "robots": [args.robots],
        "controller_configs": controller_config,
    }

    assert os.path.exists(args.bddl_file)
    problem_info = BDDLUtils.get_problem_info(args.bddl_file)
    # Check if we're using a multi-armed environment and use env_configuration argument if so

    # Create environment
    problem_name = problem_info["problem_name"]
    domain_name = problem_info["domain_name"]
    language_instruction = problem_info["language_instruction"]
    if "TwoArm" in problem_name:
        config["env_configuration"] = args.config
    env = TASK_MAPPING[problem_name](
        bddl_file_name=args.bddl_file,
        **config,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera=args.camera,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "init_data/tmp/{}_ln_{}/{}".format(
        problem_name,
        language_instruction.replace(" ", "_").strip('""'),
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = args.target_path
    os.makedirs(new_dir, exist_ok = True)

    # collect demonstrations
    remove_directory = []
    i = 0
    while i < args.num_demonstration:
        print('Collecting init states', i+1)
        saving = collect_init_states(
            env, args.arm, args.config, problem_info, remove_directory, args.bddl_file, args.video_saving
        )
        if saving:
            gather_inits(
                tmp_directory, new_dir, env_info, args.bddl_file, remove_directory
            )
            i += 1
            print('Collected demonstration ' + str(i) + '.')
