import cv2
import numpy as np
import robosuite as suite
import imageio
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from robosuite.controllers import load_controller_config


def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    return colors

segmentation_level = 'level'

def segmentation_to_rgb(seg_im, random_colors=False):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    if random_colors:
        colors = randomize_colors(N=256, bright=True)
        return (255.0 * colors[seg_im]).astype(np.uint8)
    else:
        # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
        rstate = np.random.RandomState(seed=8)
        inds = np.arange(256)
        rstate.shuffle(inds)

        # use @inds to map each geom ID to a color
        return (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]

# create environment instance
env = suite.make(
    env_name="Lift", # try with other tasks like "Stack" and "Door"
    robots="Panda",  # try with other robots like "Sawyer" and "Jaco"
    has_renderer=True,
    camera_names=["robot0_eye_in_hand"],
    camera_depths=["robot0_eye_in_hand"],
    camera_heights=[300],  # Resolution of the camera
    camera_widths=[300],
    camera_segmentations=segmentation_level,
    has_offscreen_renderer=True,
    use_camera_obs=True,
)

# reset the environment
env.reset()

env._setup_observables()

import numpy as np

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display

    image = cv2.cvtColor(obs['robot0_eye_in_hand_image'], cv2.COLOR_RGB2BGR)
    depth = obs['robot0_eye_in_hand_depth']
    seg = obs[f"robot0_eye_in_hand_segmentation_{segmentation_level}"].squeeze(-1)[::-1]
    seg = segmentation_to_rgb(seg, False)
    cv2.imshow('image', image)
    cv2.imshow('depth', depth)
    cv2.imshow('seg', seg)

    _key_ = cv2.waitKey(1)
    if _key_ == ord('q'):
        break

cv2.destroyAllWindows()