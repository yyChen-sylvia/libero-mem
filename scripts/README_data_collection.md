# Setup

## From the LIBERO root directory
1. Install LIBERO
``` bash
cd LIBERO/
pip install -e .
```

2. Install robomimic

``` bash
cd LIBERO/thirdparty/robomimic
pip install -e .
```

3. Install robosuite

``` bash
cd LIBERO/thirdparty/robosuite
pip install -e .
```

# Collect Demonstrations

## Navigate to the scripts directory:

``` bash
cd LIBERO/scripts
```

> Note: Adjust the path if necessary, depending on your folder structure.

## Run the desired command(s) to collect demonstrations:

### Example 1
``` bash
python collect_demonstration.py \
    --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE1_place_the_black_bowl_behind_the_basket_onto_the_left_plate.bddl \
    --device keyboard
```

### Example 2
``` bash
python collect_demonstration.py \
    --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE2_place_the_black_bowl_in_front_of_the_basket_onto_the_right_plate.bddl \
    --device keyboard
```

### ... and so on for each scene

Full list of commands:
``` bash
python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE1_place_the_black_bowl_behind_the_basket_onto_the_left_plate.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE2_place_the_black_bowl_in_front_of_the_basket_onto_the_right_plate.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE3_place_the_cream_cheese_box_behind_the_moka_pot_onto_the_right_bowl.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE4_place_the_cream_cheese_box_in_front_of_the_moka_pot_onto_the_left_bowl.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE5_place_the_left_bowl_on_the_right_bowl.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE6_place_the_bottle_behind_the_cabinet_onto_the_right_plate.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE7_place_the_bottle_behind_the_cabinet_to_the_left_of_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE8_place_the_bottle_behind_the_cabinet_to_the_front_of_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE9_put_the_bottle_from_the_basket_onto_the_cabinet.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE10_place_the_cream_cheese_and_the_bowl_into_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE11_place_the_cream_cheese_behind_the_cabinet_to_the_left_of_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE12_place_the_cream_cheese_behind_the_cabinet_to_the_right_of_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE13_place_the_cream_cheese_behind_the_cabinet_to_the_front_of_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE14_place_the_cream_cheese_behind_the_cabinet_to_behind_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE15_place_the_bowl_on_the_plate_behind_the_cabinet.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE16_stack_the_left_bowl_on_the_right_bowl_behind_the_cabinet.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE17_place_the_bottle_onto_the_cabinet_and_put_the_left_bowl_into_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE18_put_the_left_bowl_to_behind_the_cabinet_and_put_the_right_bowl_to_behind_the_cabinet.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE19_stack_the_left_bowl_onto_right_bowl_and_put_them_behind_the_basket.bddl --device keyboard

python collect_demonstration.py --bddl-file ../libero/libero/bddl_files/libero_relation/KITCHEN_SCENE20_stack_the_right_cream_cheese_onto_the_left_cream_cheese_and_place_them_behind_the_cabinet.bddl --device keyboard

```

### Output

- For each command, a `demo.hdf5` file will be generated in the corresponding folder under `LIBERO/scripts/demonstration_data/`.

- Make sure you have write permissions in the output directory.