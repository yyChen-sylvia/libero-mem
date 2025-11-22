"""
Collect action trajectories starting from precomputed initial states.

Usage examples:
    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_2_lift_the_bottle_and_put_it_down_on_the_plate.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_4_pick_up_the_bottle_and_put_it_down_the_plate_3_times.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_5_lift_the_bowl_and_place_it_back_on_the_plate_5_times.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_6_pick_up_the_bowl_and_place_it_on_the_plate_7_times.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center.pruned_init"

    python mem_2_collect_demonstration.py \
        --dataset-name libero_mem \
        --bddl-file "KITCHEN_SCENE1_10_put_the_cream_cheese_in_the_nearest_basket_and_place_the_empty_basket_in_the_center.bddl" \
        --init-path "__init_data/KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center.pruned_init"
"""

import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import *
from robosuite.wrappers import VisualizationWrapper
from robosuite import load_controller_config
import cv2
import numpy as np

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

from enum import Enum

class Flags(Enum):
    IS_SAVING_TRAJECTORY = 1
    IS_REPEATING_TRAJECTORY = 2
    IS_SKIPPING_INIT = 3
    IS_BACKING_INIT = 4


def save_rollout_video(rollout_images, bddl_file):
    if bddl_file is not None:
        bddl_file = os.path.basename(bddl_file)[:-5]

    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./saved_videos/"
    os.makedirs(rollout_dir, exist_ok=True)
    mp4_path = f"{rollout_dir}/{bddl_file}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path

def set_init_state(env, init_state):
    env.sim.set_state_from_flattened(init_state)
    env.sim.forward()
    env._check_success()
    env._post_process()
    env._update_observables(force=True)
    return env._get_observations()


import numpy as np

class ActionSmoother:
    def __init__(self, 
                 action_dim=7, 
                 accel_gain=0.05, 
                 decay=0.95, 
                 max_velocity=None):
        """
        Smooths pulse-like action inputs into velocity-style delta actions.
        
        Parameters:
        - action_dim (int): Dimension of the action vector (default: 7).
        - accel_gain (float): Scaling factor for how strongly pulses influence velocity.
        - decay (float): Rate at which velocity decays each timestep (0 < decay <= 1).
        - max_velocity (float or np.array): Optional max cap for each velocity dimension.
        """
        self.velocity = np.zeros(action_dim)
        self.accel_gain = accel_gain
        self.decay = decay
        self.max_velocity = max_velocity

    def reset(self):
        """Resets internal velocity state to zero."""
        self.velocity = np.zeros_like(self.velocity)

    def step(self, action_pulse):
        """
        Takes in a pulse-like action input, returns a smoothed delta-action.
        
        Parameters:
        - action_pulse (array-like): A 7D action vector (usually 0s and 1s or -1s).
        
        Returns:
        - smoothed_action (np.array): Smoothed delta-action to send to the robot.
        """
        action_pulse = np.array(action_pulse)
        
        # Update velocity with acceleration from the pulse
        self.velocity = self.decay * self.velocity + self.accel_gain * action_pulse

        # Optionally clamp velocity
        if self.max_velocity is not None:
            self.velocity = np.clip(self.velocity, 
                                    -np.abs(self.max_velocity), 
                                     np.abs(self.max_velocity))
        
        # Copy the gripping action
        self.velocity[-1] = action_pulse[-1]

        return self.velocity.copy()

def collect_human_trajectory(
    env, init_dict, init_state, device, arm, env_configuration, problem_info, remove_directory=[], bddl_file=None, video_saving=False
):
    """
    Use the device (keyboard or SpaceNav 3D mouse) to collect a demonstration.
    The rollout trajectory is saved to files in npz format.
    Modify the DataCollectionWrapper wrapper to add new fields or change data formats.

    Args:
        env (MujocoEnv): environment to control
        init_state (tensor): init states of the objects
        device (Device): to receive controls from the device
        arms (str): which arm to control (eg bimanual) 'right' or 'left'
        env_configuration (str): specified environment configuration
    """

    env.reset()
    set_init_state(env, init_state)
    env.reset_subgoal_progress()

    smoother = ActionSmoother(accel_gain=0.05, decay=0.60, max_velocity=0.4)

    # ID = 2 always corresponds to agentview
    # env.render()

    task_completion_hold_count = (
        -1
    )  # counter to collect 10 timesteps after reaching goal
    device.start_control()

    # Loop until we get a reset from the input or the task completes
    saving = Flags.IS_SAVING_TRAJECTORY
    count = 0
    success_count = 0
    rollout_video = []
    while True:
        count += 1
        # Set active robot
        active_robot = (
            env.robots[0]
            if env_configuration == "bimanual"
            else env.robots[arm == "left"]
        )

        # Get the newest action
        action, grasp = input2action(
            device=device,
            robot=active_robot,
            active_arm=arm,
            env_configuration=env_configuration,
        )

        # If action is none, then this a reset so we should break
        if action is None:
            while True:
                _key_ = cv2.waitKey(1)
                if _key_ == ord('q'):    
                    print("Break")
                    saving = Flags.IS_REPEATING_TRAJECTORY
                    break
                elif _key_ == ord('i'):
                    print("Skip init")
                    saving = Flags.IS_SKIPPING_INIT
                    break
                elif _key_ == ord('s'):
                    print("Save trajectory")
                    saving = Flags.IS_SAVING_TRAJECTORY
                    break
                elif _key_ == ord('r'):
                    print("Back trajectory")
                    saving = Flags.IS_BACKING_INIT
                    break
            break

        # Run environment step
        action = smoother.step(action)
        # print(action)
        obs, reward, done, info = env.step(action)
        image = env.render()
        # print('')
        ego_view = obs['robot0_eye_in_hand_image'][:,::-1]
        exo_view = obs['agentview_image'][::-1,:]
        camera_view = cv2.hconcat([exo_view, ego_view])

        cv2.imshow('camera_view', cv2.cvtColor(camera_view, cv2.COLOR_RGB2BGR))
        if video_saving:
            rollout_video.append(camera_view)

        # state machine to check for having a success for 10 consecutive timesteps
        if env._check_success(inc=True):
            if task_completion_hold_count < 0:
                task_completion_hold_count = 10  # reset count on first success timestep
                success_count += 1
            else:
                task_completion_hold_count = 10
        else:
            task_completion_hold_count -= 1

            # task_completion_hold_count = -10  # null the counter if there's no success

        print("The done jobs are,", env.get_satisfied_subgoals())
        if success_count > 0:
            while True:
                _key_ = cv2.waitKey(1)
                if _key_ == ord('q'):    
                    print("Break")
                    saving = Flags.IS_REPEATING_TRAJECTORY
                    break
                elif _key_ == ord('i'):
                    print("Skip init")
                    saving = Flags.IS_SKIPPING_INIT
                    break
                elif _key_ == ord('s'):
                    print("Save trajectory")
                    saving = Flags.IS_SAVING_TRAJECTORY
                    break
            break

    if video_saving:
        save_rollout_video(rollout_video, bddl_file); 1/0

    # cleanup for end of data collection episodes
    if saving != Flags.IS_SAVING_TRAJECTORY:
        remove_directory.append(env.ep_directory.split("/")[-1])
    init_dict[env.ep_directory.split("/")[-1]] = init_state
    env.close()
    return saving


def gather_demonstrations_as_hdf5(
    time_ext, directory, out_dir, init_dict, env_info, args, remove_directory=[]
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
        init_dict (dict): Initialization dictionary of state path and state
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info
    """
    demo_name = args.bddl_file.split('/')[-1][:-5] + '_demo' + time_ext + '.hdf5'
    hdf5_path = os.path.join(out_dir, demo_name)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = 0
    env_name = None  # will get populated at some point

    directory_list = os.listdir(directory)
    directory_list.sort()
    for ep_directory in directory_list:
        # print(ep_directory)
        if ep_directory in remove_directory:
            # print("Skipping")
            continue
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        # Delete the first actions and the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group("demo_{}".format(num_eps))

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as f:
            xml_str = f.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("init_state", data=init_dict[ep_directory])
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = datetime.datetime.now()
    grp.attrs["date"] = "{}-{}-{}".format(now.month, now.day, now.year)
    grp.attrs["time"] = "{}:{}:{}".format(now.hour, now.minute, now.second)
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    grp.attrs["problem_info"] = json.dumps(problem_info)
    grp.attrs["bddl_file_name"] = args.bddl_file
    grp.attrs["bddl_file_content"] = str(open(args.bddl_file, "r", encoding="utf-8"))

    f.close()


def get_camera_param(env, camera_name):
    # Retrieve camera ID from the model
    camera_id = env.sim.model.camera_name2id(camera_name)

    # Get vertical field of view (in degrees)
    fovy = env.sim.model.cam_fovy[camera_id]

    image_width = 256
    image_height = 256

    # Convert fovy from degrees to radians
    fovy_rad = np.deg2rad(fovy)

    # Compute focal length in pixels
    f_y = image_height / (2 * np.tan(fovy_rad / 2))
    f_x = f_y  # Assuming square pixels

    # Principal point (cx, cy) at the image center
    c_x = image_width / 2
    c_y = image_height / 2

    # Intrinsic camera matrix (K)
    K = np.array([
        [f_x, 0, c_x],
        [0, f_y, c_y],
        [0, 0, 1]
    ])

    # Display the intrinsic matrix
    print("Focal Length (fx, fy):", (f_x, f_y))
    print("Principal Point (cx, cy):", (c_x, c_y))
    print("Intrinsic Camera Matrix K:\n", K)

import torch
def get_task_init_states(init_states_path):
    init_states = torch.load(init_states_path)
    return init_states


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="demonstration_data",
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
    parser.add_argument("--device", type=str, default="keyboard")
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
        help="How much demonstrations to perform",
    )
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--bddl-file", type=str)
    parser.add_argument("--init-path", type=str)

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

    get_camera_param(env, camera_name='agentview')
    get_camera_param(env, camera_name='robot0_eye_in_hand')

    # Wrap this with visualization wrapper
    env = VisualizationWrapper(env)

    # Grab reference to controller config and convert it to json-encoded string
    env_info = json.dumps(config)

    # wrap the environment with data collection wrapper
    tmp_directory = "demonstration_data/tmp/{}/{}".format(
        args.bddl_file.split('/')[-1][:-5],
        str(time.time()).replace(".", "_"),
    )

    env = DataCollectionWrapper(env, tmp_directory)


    # initialize device
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(
            pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity
        )
        env.viewer.add_keypress_callback(device.on_press) # "any", 
        # env.viewer.add_keyup_callback(device.on_release) # "any", 
        # env.viewer.add_keyrepeat_callback(device.on_press) # "any", 
    elif args.device == "spacemouse":
        from robosuite.devices import SpaceMouse

        device = SpaceMouse(
            args.vendor_id,
            args.product_id,
            pos_sensitivity=args.pos_sensitivity,
            rot_sensitivity=args.rot_sensitivity,
        )
    else:
        raise Exception(
            "Invalid device choice: choose either 'keyboard' or 'spacemouse'."
        )

    # make a new timestamped directory
    t1, t2 = str(time.time()).split(".")
    new_dir = os.path.join(
        args.directory,
        f"{args.dataset_name}"
    )
    time_ext = f'_{t1}_{t2}'

    os.makedirs(new_dir, exist_ok=True)

    # collect demonstrations
    init_states_path = args.init_path
    init_states = get_task_init_states(init_states_path)
    init_dict = {}
    remove_directory = []
    i = 0
    init_id = 0
    while i < args.num_demonstration and init_id < len(init_states):
        print('Collecting demonstration', i+1)

        key = collect_human_trajectory(
            env, init_dict, init_states[init_id], device, args.arm, args.config, problem_info, remove_directory, args.bddl_file, args.video_saving
        )
        print(remove_directory)
        if key == Flags.IS_SAVING_TRAJECTORY:
            gather_demonstrations_as_hdf5(
                time_ext, tmp_directory, new_dir, init_dict, env_info, args, remove_directory
            )
            i += 1
            init_id += 1
            print('Collected demonstration ' + str(i) + '.')

        if key == Flags.IS_SKIPPING_INIT:
            init_id += 1
        if key == Flags.IS_BACKING_INIT:
            init_id -= 1

