"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig, SlotVLAV2Config, CustomOpenVLAConfig, ObjectCentricVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction, EmbodiedDecodedSlotSSM, EmbodiedDecodedSlotSSMv2
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

from peft import PeftModel
from safetensors.torch import load_file

def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    # print("[*] Loading in F16 with Flash-Attention Enabled")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    if 'libero' in cfg.pretrained_checkpoint:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        adapter_dir = cfg.pretrained_checkpoint.replace('/ckpts/', '/adapters/')

        base_vla = AutoModelForVision2Seq.from_pretrained(
            "/home/nhatcm3/workspace/robot-slotssm/output_hf_model_openx", 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to('cuda')
        print("Loaded base.")

        base_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        base_vla = base_vla.merge_and_unload()
        print("Merged CustomOpenVLA LLM LoRA.")

        if '--noslotgoal' in cfg.pretrained_checkpoint:
            print("Not using slot goal.")
            vla = EmbodiedDecodedSlotSSMv2(base_model=base_vla, use_slotgoals=False)
        else:
            print("Using slot goal.")
            vla = EmbodiedDecodedSlotSSMv2(base_model=base_vla)
        # weights = load_file(os.path.join(adapter_dir, 'object_centric_forward_actionable.safetensors'))
        weights = load_file(os.path.join(adapter_dir, 'object_centric_backward_actionable.safetensors'))
        vla.load_state_dict(weights, strict=False)
        vla.requires_grad_(False)
        vla = vla.to('cuda')
        print("Loaded CustomOpenVLA additional base.")

        vla = vla
    else:
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        if 'multiview_dsx8' in cfg.pretrained_checkpoint:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPredictionDSX8)

        elif 'multiview_ds' in cfg.pretrained_checkpoint:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPredictionDS)

        else:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        adapter_dir = cfg.pretrained_checkpoint.replace('/ckpts/', '/adapters/')
        base_vla = AutoModelForVision2Seq.from_pretrained(
            "output_hf_model_openx", 
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            # torch_dtype=torch.float16,
            load_in_8bit=cfg.load_in_8bit,
            load_in_4bit=cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        ).to('cuda')
        print("Loaded base:", type(base_vla))
        base_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        base_vla = base_vla.merge_and_unload()
        vla = base_vla

        # vla = AutoModelForVision2Seq.from_pretrained(
        #     cfg.pretrained_checkpoint,
        #     attn_implementation="flash_attention_2",
        #     torch_dtype=torch.bfloat16,
        #     load_in_8bit=cfg.load_in_8bit,
        #     load_in_4bit=cfg.load_in_4bit,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=False,
        # )
        # print("Loaded vla model base.")


    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=False)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut: _ _ _ _ _ _ _ _</s>"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.float16)
    inputs['input_ids'][0][-9] = 29871
    # print(inputs['input_ids']); 1/0

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def get_vla_action_v2(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    images = [obs["full_image"], obs["wrist_image"]]
    input_data = []
    for image in images:
        image = Image.fromarray(image)
        image = image.convert("RGB")

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Crop and then resize back to original size
            image = crop_and_resize(image, crop_scale, batch_size)

            # Convert back to original data type
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            image = Image.fromarray(image.numpy())
            image = image.convert("RGB")
        
        # Build VLA prompt
        if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
            prompt = (
                f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
            )
        else:  # OpenVLA
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut: _ _ _ _ _ _ _ _</s>"

        # Process inputs.
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.float16)
        inputs['input_ids'][0][-9] = 29871
        input_data.append(inputs)

    # Get action.
    inputs = {}
    inputs['pixel_values'] = []
    for datum in input_data:
        inputs['input_ids'] = datum['input_ids']
        inputs['attention_mask'] = datum['attention_mask']
        bz, cc, h, w = datum['pixel_values'].shape
        datum['pixel_values'] = datum['pixel_values'].reshape(bz, 1, cc, h, w)
        inputs['pixel_values'].append(datum['pixel_values'])
    inputs['pixel_values'] = torch.stack(inputs['pixel_values'], dim=1)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action


def get_vla_action_v3(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    images = [obs["full_image"], obs["wrist_image"], obs["depth_full_image"], obs["depth_wrist_image"]]
    input_data = []
    for image in images:
        image = Image.fromarray(image)
        image = image.convert("RGB")

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Crop and then resize back to original size
            image = crop_and_resize(image, crop_scale, batch_size)

            # Convert back to original data type
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            image = Image.fromarray(image.numpy())
            image = image.convert("RGB")
        
        # Build VLA prompt
        if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
            prompt = (
                f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
            )
        else:  # OpenVLA
            prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        # Process inputs.
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
        input_data.append(inputs)

    # Get action.
    inputs = {}
    inputs['pixel_values'] = []
    for datum in input_data:
        inputs['input_ids'] = datum['input_ids']
        inputs['attention_mask'] = datum['attention_mask']
        bz, cc, h, w = datum['pixel_values'].shape
        datum['pixel_values'] = datum['pixel_values'].reshape(bz, 1, cc, h, w)
        inputs['pixel_values'].append(datum['pixel_values'])
    inputs['pixel_values'] = torch.stack(inputs['pixel_values'], dim=1)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action
