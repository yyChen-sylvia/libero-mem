"""
Merge demonstrations from a relabeled LIBERO dataset (HDF5 files) into a
consolidated format.

Usage examples:
    python mem_4_merge_demonstrations.py \
        --libero_task_suite libero_goal \
        --libero_raw_data_dir ./libero_goal/revised/ \
        --libero_target_dir ./libero_goal/revised_combined/

    python mem_4_merge_demonstrations.py \
        --libero_task_suite libero_mem \
        --libero_raw_data_dir path/to/libero_mem_revised \
        --libero_target_dir path/to/libero_mem_combined
"""

import argparse
import json
import os
import time

import cv2
import h5py
import numpy as np
import tqdm
from libero.libero import benchmark

import sys
sys.path.append('/home/nhatchung/Desktop/dataspace/libero/LIBERO')

def sort_key_hdf5(name):
    # extract number after 'demo_'
    number = int(name.split('_')[-1].split('.')[0])
    return number

def sort_key_metainfo(name):
    # extract number after 'demo_'
    number = int(name.split('_')[-2].split('.')[0])
    return number

def recursive_merge(dest, src):
    for key, value in src.items():
        if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
            recursive_merge(dest[key], value)
        else:
            dest[key] = value

def recursive_copy(src, dest):
    for key in src.keys():
        if isinstance(src[key], h5py.Group):
            new_grp = dest.create_group(key)
            recursive_copy(src[key], new_grp)
        elif isinstance(src[key], h5py.Dataset):
            src.copy(key, dest)
    for attr_key in src.attrs:
        dest.attrs[attr_key] = src.attrs[attr_key]

def main(args):
    print(f"Regenerating {args.libero_task_suite} dataset!")
    os.makedirs(args.libero_target_dir, exist_ok=True)

    # Prepare JSON file to record success/false and initial states per episode
    metainfo_json_dict = {}
    metainfo_json_out_path = os.path.join(args.libero_target_dir, f"./{args.libero_task_suite}_metainfo.json")
    with open(metainfo_json_out_path, "w") as f:
        # Just test that we can write to this file (we overwrite it later)
        json.dump(metainfo_json_dict, f)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.libero_task_suite]()
    num_tasks_in_suite = task_suite.n_tasks
    
    # Iterate through the task suites
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task in suite
        task = task_suite.get_task(task_id)
        data_dir = os.path.join(args.libero_raw_data_dir, task.name)
        data_files = os.listdir(data_dir)
        
        hdf5_files = [_file for _file in data_files if '.hdf5' in _file]
        hdf5_files = sorted(hdf5_files, key=sort_key_hdf5)
        meta_files = [_file for _file in data_files if '_metainfo.json' in _file]
        meta_files = sorted(meta_files, key=sort_key_metainfo)

        # Create new HDF5 file for regenerated demos
        new_data_path = os.path.join(args.libero_target_dir, f"{task.name}_demo.hdf5")
        new_data_file = h5py.File(new_data_path, "w")
        grp = new_data_file.create_group("data")

        for idx, hdf5_name in tqdm.tqdm(enumerate(hdf5_files)):
            hdf5_name = os.path.join(data_dir, hdf5_name)
            traj_data_file = h5py.File(hdf5_name, "r")
            traj_data = traj_data_file["data"]

            # Copy trajectory data
            for ep_key in traj_data.keys():
                src_grp = traj_data[ep_key]
                dest_grp = grp.create_group(ep_key)
                recursive_copy(src_grp, dest_grp)
            
            traj_data_file.close()

            meta_name = os.path.join(data_dir, meta_files[idx])
            with open(meta_name, "r") as f:
                # Just test that we can write to this file (we overwrite it later)
                meta_data = json.load(f)
                meta_data_key = list(meta_data.keys())[0]
                demo_data_key = list(meta_data[meta_data_key].keys())[0]
                indexed_meta_data = meta_data[meta_data_key][demo_data_key]

                # For each type of boxes
                for boxes_key in ['exo_boxes', 'ego_boxes']:
                    # For each object dictionary
                    for datum in indexed_meta_data[boxes_key]:
                        for datum_key, datum_value in datum.items():
                            # For every plate
                            if 'plate' in datum_key:
                                datum_value[-1] = 0
                                datum[datum_key] = datum_value

                # Recursively merge the meta data
                recursive_merge(metainfo_json_dict, meta_data)

                # Write metainfo dict to JSON file
                # (We repeatedly overwrite, rather than doing this once at the end, just in case the script crashes midway)
                with open(metainfo_json_out_path, "w") as f:
                    json.dump(metainfo_json_dict, f, indent=2)

            # if idx > 1:
            #     break
        
        new_data_file.close()
        # 1/0

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero_task_suite", type=str, choices=["libero_mem", "libero_relation", "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
                        help="LIBERO task suite. Example: libero_spatial", required=True)
    parser.add_argument("--libero_raw_data_dir", type=str,
                        help="Path to directory containing raw HDF5 dataset. Example: ./LIBERO/libero/datasets/libero_spatial", required=True)
    parser.add_argument("--libero_target_dir", type=str,
                        help="Path to regenerated dataset directory. Example: ./LIBERO/libero/datasets/libero_spatial_no_noops", required=True)
    args = parser.parse_args()

    main(args)