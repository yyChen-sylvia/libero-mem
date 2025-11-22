from robosuite.utils.mjcf_utils import new_site

from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Libero_Kitchen_Tabletop_Manipulation(BDDLBaseDomain):
    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "kitchen_table"
        self.visualization_sites_list = []
        if "table_full_size" in kwargs:
            self.kitchen_table_full_size = table_full_size
        else:
            self.kitchen_table_full_size = (1.0, 1.2, 0.05)
        self.kitchen_table_offset = (0.0, 0, 0.90)
        # For z offset of environment fixtures
        self.z_offset = 0.01 - self.kitchen_table_full_size[2]
        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.kitchen_table_offset})
        kwargs.update({"arena_type": "kitchen"})
        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update(
                {"scene_xml": "scenes/libero_kitchen_tabletop_base_style.xml"}
            )
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "gray-ceramic",
                        "wall_style": "yellow-linen",
                    }
                }
            )

        super().__init__(bddl_file_name, *args, **kwargs)

    def _load_fixtures_in_arena(self, mujoco_arena):
        """Nothing extra to load in this simple problem."""
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "kitchen_table":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        # Create site objects
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):

            if "kitchen_table" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2 + self.workspace_offset[0],
                    (ranges[3] + ranges[1]) / 2 + self.workspace_offset[1],
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    z_offset=self.workspace_offset[2],
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone
                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos + np.array([0.0, 0.0, -0.90]),
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            # Otherwise the processing is consistent
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            # This is a special case for CompositeObject, we skip this as this is very rare in our benchmark
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                object_sites_dict[object_region_name] = SiteObject(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    size=site.get("size"),
                                    rgba=site.get("rgba"),
                                    site_type=site.get("type"),
                                    site_pos=site.get("pos"),
                                    site_quat=site.get("quat"),
                                    object_properties=body.object_properties,
                                )
        self.object_sites_dict = object_sites_dict

        # Keep track of visualization objects
        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _add_placement_initializer(self):
        """Very simple implementation at the moment. Will need to upgrade for other relations later."""
        super()._add_placement_initializer()

    def get_goal_sequence_len(self):
        all_goals = self.get_all_goals()
        return len(all_goals[0])

    def get_all_goals(self):
        goal_state = self.parsed_problem["goal_state"]
        if goal_state[0][0] == 'or':
            goal_state = goal_state[0]
            return self._get_goals(goal_state)
        elif goal_state[0][0] == 'sequence':
            goal_state = goal_state[0]
            return [self._get_goals(goal_state)]
        else:
            return [[goal_state]] # 1 sequence of length 1
        
    def _get_goals(self, goal_state):
        op = goal_state[0]
        if op == "and":
            subgoal = goal_state[1:]
            return subgoal
        
        elif op == "sequence":
            goal_sequence = []
            subgoals = goal_state[1:]
            for subgoal in subgoals:
                goal_sequence.append(self._get_goals(subgoal))
            return goal_sequence

        elif op == "or":
            goal_sequences = []
            subsequences = goal_state[1:]
            for subsequence in subsequences:
                goal_sequences.append(self._get_goals(subsequence))
            return goal_sequences


    def _check_success(self, inc=False):
        """
        Check if the goal is achieved. Handles nested 'and', 'or', and 'sequence' structures.
        """
        goal_state = self.parsed_problem["goal_state"]
        is_leaf_level = (goal_state[0][0] != 'sequence') and (goal_state[0][0] != 'or')
        if not is_leaf_level:
            goal_state = goal_state[0]
        return self._eval_goal(goal_state, inc, is_leaf_level=is_leaf_level)

    def _eval_goal(self, goal_state, inc=False, is_leaf_level=False):
        """
        Recursively evaluate goal structures.
        Supports: ['and', ...], ['or', ...], ['sequence', ...], or primitive predicates.
        """
        if is_leaf_level:
            result = True
            for state in goal_state:
                result = self._eval_predicate(state) and result                
            return result

        op = goal_state[0]
        if op == "and":
            subgoal = goal_state[1:]
            if not self._eval_goal(subgoal, inc, is_leaf_level=True):
                return False
            return True

        elif op == "sequence":
            idx = len(self._satisfied_subgoals)
            if self._overshot: # if overshooting has happened then it will forever be false
                return False 

            if idx >= len(goal_state) - 1:
                # here we check for overshooting if last goal state no longer hold true (then it will only be False afterwards)
                idx = idx - 1

            subgoal = goal_state[idx + 1]  # offset by 1 because goal[0] == 'sequence'
            assert(len(subgoal[1:]) == 1)  # only has 1 subgoal state for 'and' op
            success = self._eval_goal(subgoal, inc, is_leaf_level=is_leaf_level)

            if success:
                if len(self._satisfied_subgoals) >= len(goal_state) - 1: # success as always
                    self._sub_goal_nonlive_time = 0 # reset final goal counter to check overshoot and small deviations
                    return True
                
                if self._sub_goal_live_time > 5 and self._sub_goal_nonlive_time > 15:
                    self._sub_goal_live_time = 0
                    self._sub_goal_nonlive_time = 0
                    self._satisfied_subgoals.append(subgoal)
                    return True
                else:
                    if inc:
                        self._sub_goal_live_time += 1
            else:
                if len(self._satisfied_subgoals) >= len(goal_state) - 1: # success has changed state
                    if self._sub_goal_nonlive_time > 5:
                        self._overshot = True
                        return False

                if inc:
                    self._sub_goal_nonlive_time += 1

            return False

        elif op == "or":
            for subgoal in goal_state[1:]:
                
                # check if this subgoal trajectory matches the prefix of satisfied subgoals
                assert(subgoal[0] == 'sequence')
                if not self.is_subgoal_trajectory_possible(current_traj=self._satisfied_subgoals,
                                                      testing_traj=subgoal[1:]):
                    continue

                if self._eval_goal(subgoal, inc, is_leaf_level=is_leaf_level):
                    return True

            return False

        else:
            raise ValueError(f"Unknown goal structure: {goal_state}")

    # def _check_success(self, inc=False):
    #     """
    #     Check if the goal is achieved. Consider conjunction goals at the moment
    #     """
    #     goal_state = self.parsed_problem["goal_state"]
    #     print(goal_state); 1/0
    #     if goal_state[0][0] == 'sequence':
    #         if self._sub_goal_idx >= len(goal_state[0]):
    #             result = True
            
    #         else:
    #             sub_goal_state = goal_state[0][self._sub_goal_idx][1:]
    #             if inc:
    #                 print(sub_goal_state)
    #             result = True
    #             for state in sub_goal_state:
    #                 result = self._eval_predicate(state) and result
                
    #             if result:
    #                 if self._sub_goal_live_time > 5 and self._sub_goal_nonlive_time > 15:
    #                     self._sub_goal_idx += 1
    #                     self._sub_goal_live_time = 0
    #                     self._sub_goal_nonlive_time = 0
    #                 else:
    #                     if inc:
    #                         self._sub_goal_live_time += 1
    #             else:
    #                 if inc:
    #                     self._sub_goal_nonlive_time += 1
                
    #             # if inc:
    #             #     print(sub_goal_state, self._sub_goal_live_time, result)
    #             result = result and (self._sub_goal_idx >= len(goal_state[0]))
    #     else:
    #         result = True
    #         for state in goal_state:
    #             result = self._eval_predicate(state) and result
    #     return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            # Checking binary logical predicates
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            # Checking unary logical predicates
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _setup_references(self):
        super()._setup_references()

    def _post_process(self):
        super()._post_process()

        self.set_visualization()

    def set_visualization(self):

        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    # We toggle the alpha value
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.6586131746834771, 0.0, 1.6103500240372423],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )

        # For visualization purpose
        mujoco_arena.set_camera(
            camera_name="frontview", pos=[1.0, 0.0, 1.48], quat=[0.56, 0.43, 0.43, 0.56]
        )
        mujoco_arena.set_camera(
            camera_name="galleryview",
            pos=[2.844547668904445, 2.1279684793440667, 3.128616846013882],
            quat=[
                0.42261379957199097,
                0.23374411463737488,
                0.41646939516067505,
                0.7702690958976746,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="paperview",
            pos=[2.1, 0.535, 2.075],
            quat=[0.513, 0.353, 0.443, 0.645],
        )
