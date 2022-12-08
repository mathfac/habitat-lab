#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import os.path as osp
import threading
from functools import partial
from typing import List, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from habitat.config.default_structured_configs import (
    HabitatConfigPlugin,
    register_hydra_plugin,
)
from habitat.config.read_write import read_write

_HABITAT_CFG_DIR = osp.dirname(inspect.getabsfile(inspect.currentframe()))
# Habitat config directory inside the installed package.
# Used to access default predefined configs.
# This is equivalent to doing osp.dirname(osp.abspath(__file__))
# in editable install, this is pwd/habitat-lab/habitat/config
CONFIG_FILE_SEPARATOR = ","

<<<<<<< HEAD
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.habitat = CN()
_C.habitat.seed = 100
# -----------------------------------------------------------------------------
# environment
# -----------------------------------------------------------------------------
_C.habitat.environment = CN()
_C.habitat.environment.max_episode_steps = 1000
_C.habitat.environment.max_episode_seconds = 10000000
_C.habitat.environment.iterator_options = CN()
_C.habitat.environment.iterator_options.cycle = True
_C.habitat.environment.iterator_options.shuffle = True
_C.habitat.environment.iterator_options.group_by_scene = True
_C.habitat.environment.iterator_options.num_episode_sample = -1
_C.habitat.environment.iterator_options.max_scene_repeat_episodes = -1
_C.habitat.environment.iterator_options.max_scene_repeat_steps = int(1e4)
_C.habitat.environment.iterator_options.step_repetition_range = 0.2
# -----------------------------------------------------------------------------
# task
# -----------------------------------------------------------------------------
_C.habitat.task = CN()
_C.habitat.task.reward_measure = None
_C.habitat.task.success_measure = None
_C.habitat.task.success_reward = 2.5
_C.habitat.task.slack_reward = -0.01
_C.habitat.task.end_on_success = False
# -----------------------------------------------------------------------------
# # NAVIGATION task
# -----------------------------------------------------------------------------
_C.habitat.task.type = "Nav-v0"
_C.habitat.task.sensors = []
_C.habitat.task.measurements = []
_C.habitat.task.goal_sensor_uuid = "pointgoal"
_C.habitat.task.possible_actions = [
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
]
# -----------------------------------------------------------------------------
# # REARRANGE task
# -----------------------------------------------------------------------------
_C.habitat.task.count_obj_collisions = True
_C.habitat.task.settle_steps = 5
_C.habitat.task.constraint_violation_ends_episode = True
_C.habitat.task.constraint_violation_drops_object = False
_C.habitat.task.force_regenerate = (
    False  # Forced to regenerate the starts even if they are already cached.
)
_C.habitat.task.should_save_to_cache = False  # Saves the generated starts to a cache if they are not already generated.
_C.habitat.task.must_look_at_targ = True
_C.habitat.task.object_in_hand_sample_prob = 0.167
_C.habitat.task.gfx_replay_dir = "data/replays"
_C.habitat.task.render_target = True
_C.habitat.task.ee_sample_factor = 0.2
_C.habitat.task.ee_exclude_region = 0.0
# In radians
_C.habitat.task.base_angle_noise = 0.15
_C.habitat.task.base_noise = 0.05
_C.habitat.task.spawn_region_scale = 0.2
_C.habitat.task.joint_max_impulse = -1.0
_C.habitat.task.desired_resting_position = [0.5, 0.0, 1.0]
_C.habitat.task.use_marker_t = True
_C.habitat.task.cache_robot_init = False
_C.habitat.task.success_state = 0.0
# If true, does not care about navigability or collisions with objects when spawning
# robot
_C.habitat.task.easy_init = False
_C.habitat.task.should_enforce_target_within_reach = False
# -----------------------------------------------------------------------------
# # COMPOSITE task CONFIG
# -----------------------------------------------------------------------------
_C.habitat.task.task_spec_base_path = "tasks/rearrange/pddl/"
_C.habitat.task.task_spec = ""
# PDDL domain params
_C.habitat.task.pddl_domain_def = "replica_cad"
_C.habitat.task.obj_succ_thresh = 0.3
_C.habitat.task.art_succ_thresh = 0.15
_C.habitat.task.robot_at_thresh = 2.0
_C.habitat.task.filter_nav_to_tasks = []
# -----------------------------------------------------------------------------
# # actions
# -----------------------------------------------------------------------------
_C.habitat.task.actions = CN()
_C.habitat.task.actions.stop = CN()
_C.habitat.task.actions.stop.type = "StopAction"
_C.habitat.task.actions.empty = CN()
_C.habitat.task.actions.empty.type = "EmptyAction"
# -----------------------------------------------------------------------------
# # NAVIGATION actions
# -----------------------------------------------------------------------------
_C.habitat.task.actions.move_forward = CN()
_C.habitat.task.actions.move_forward.type = "MoveForwardAction"
_C.habitat.task.actions.turn_left = CN()
_C.habitat.task.actions.turn_left.type = "TurnLeftAction"
_C.habitat.task.actions.turn_right = CN()
_C.habitat.task.actions.turn_right.type = "TurnRightAction"
_C.habitat.task.actions.look_up = CN()
_C.habitat.task.actions.look_up.type = "LookUpAction"
_C.habitat.task.actions.look_down = CN()
_C.habitat.task.actions.look_down.type = "LookDownAction"
_C.habitat.task.actions.teleport = CN()
_C.habitat.task.actions.teleport.type = "TeleportAction"
_C.habitat.task.actions.velocity_control = CN()
_C.habitat.task.actions.velocity_control.type = "VelocityAction"
_C.habitat.task.actions.velocity_control.lin_vel_range = [
    0.0,
    0.25,
]  # meters per sec
_C.habitat.task.actions.velocity_control.ang_vel_range = [
    -10.0,
    10.0,
]  # deg per sec
_C.habitat.task.actions.velocity_control.min_abs_lin_speed = (
    0.025  # meters per sec
)
_C.habitat.task.actions.velocity_control.min_abs_ang_speed = 1.0  # deg per sec
_C.habitat.task.actions.velocity_control.time_step = 1.0  # seconds
# -----------------------------------------------------------------------------
# # REARRANGE actions
# -----------------------------------------------------------------------------
_C.habitat.task.actions.arm_action = CN()
_C.habitat.task.actions.arm_action.type = "ArmAction"
_C.habitat.task.actions.arm_action.arm_controller = "ArmRelPosAction"
_C.habitat.task.actions.arm_action.grip_controller = None
_C.habitat.task.actions.arm_action.arm_joint_dimensionality = 7
_C.habitat.task.actions.arm_action.grasp_thresh_dist = 0.15
_C.habitat.task.actions.arm_action.disable_grip = False
_C.habitat.task.actions.arm_action.delta_pos_limit = 0.0125
_C.habitat.task.actions.arm_action.ee_ctrl_lim = 0.015
_C.habitat.task.actions.arm_action.should_clip = False
_C.habitat.task.actions.arm_action.render_ee_target = False
_C.habitat.task.actions.arm_action.agent = None
_C.habitat.task.actions.base_velocity = CN()
_C.habitat.task.actions.base_velocity.type = "BaseVelAction"
_C.habitat.task.actions.base_velocity.lin_speed = 10.0
_C.habitat.task.actions.base_velocity.ang_speed = 10.0
_C.habitat.task.actions.base_velocity.allow_dyn_slide = True
_C.habitat.task.actions.base_velocity.end_on_stop = False
_C.habitat.task.actions.base_velocity.allow_back = True
_C.habitat.task.actions.base_velocity.min_abs_lin_speed = 1.0
_C.habitat.task.actions.base_velocity.min_abs_ang_speed = 1.0
_C.habitat.task.actions.base_velocity.agent = None
_C.habitat.task.actions.rearrange_stop = CN()
_C.habitat.task.actions.rearrange_stop.type = "RearrangeStopAction"
# -----------------------------------------------------------------------------
# Oracle navigation action
# This action takes as input a discrete ID which refers to an object in the
# PDDL domain. The oracle navigation controller then computes the actions to
# navigate to that desired object.
# -----------------------------------------------------------------------------
_C.habitat.task.actions.oracle_nav_action = CN()
_C.habitat.task.actions.oracle_nav_action.type = "OracleNavAction"
_C.habitat.task.actions.oracle_nav_action.turn_velocity = 1.0
_C.habitat.task.actions.oracle_nav_action.forward_velocity = 1.0
_C.habitat.task.actions.oracle_nav_action.turn_thresh = 0.1
_C.habitat.task.actions.oracle_nav_action.dist_thresh = 0.2
_C.habitat.task.actions.oracle_nav_action.agent = None
_C.habitat.task.actions.oracle_nav_action.lin_speed = 10.0
_C.habitat.task.actions.oracle_nav_action.ang_speed = 10.0
_C.habitat.task.actions.oracle_nav_action.min_abs_lin_speed = 1.0
_C.habitat.task.actions.oracle_nav_action.min_abs_ang_speed = 1.0
_C.habitat.task.actions.oracle_nav_action.allow_dyn_slide = True
_C.habitat.task.actions.oracle_nav_action.end_on_stop = False
_C.habitat.task.actions.oracle_nav_action.allow_back = True
# -----------------------------------------------------------------------------
# # TASK_SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.pointgoal_sensor = CN()
_C.habitat.task.pointgoal_sensor.type = "PointGoalSensor"
_C.habitat.task.pointgoal_sensor.goal_format = "POLAR"
_C.habitat.task.pointgoal_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.pointgoal_with_gps_compass_sensor = (
    _C.habitat.task.pointgoal_sensor.clone()
)
_C.habitat.task.pointgoal_with_gps_compass_sensor.type = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.OBJECTgoal_sensor = CN()
_C.habitat.task.OBJECTgoal_sensor.type = "ObjectGoalSensor"
_C.habitat.task.OBJECTgoal_sensor.goal_spec = "TASK_CATEGORY_ID"
_C.habitat.task.OBJECTgoal_sensor.goal_spec_max_val = 50
# -----------------------------------------------------------------------------
# IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.imagegoal_sensor = CN()
_C.habitat.task.imagegoal_sensor.type = "ImageGoalSensor"
# -----------------------------------------------------------------------------
# INSTANCE IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.instance_imagegoal_sensor = CN()
_C.habitat.task.instance_imagegoal_sensor.type = "InstanceImageGoalSensor"
# -----------------------------------------------------------------------------
# INSTANCE IMAGEGOAL HFOV SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.instance_imagegoal_hfov_sensor = CN()
_C.habitat.task.instance_imagegoal_hfov_sensor.type = (
    "InstanceImageGoalHFOVSensor"
)
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.heading_sensor = CN()
_C.habitat.task.heading_sensor.type = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.compass_sensor = CN()
_C.habitat.task.compass_sensor.type = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.gps_sensor = CN()
_C.habitat.task.gps_sensor.type = "GPSSensor"
_C.habitat.task.gps_sensor.dimensionality = 2
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.proximity_sensor = CN()
_C.habitat.task.proximity_sensor.type = "ProximitySensor"
_C.habitat.task.proximity_sensor.max_detection_radius = 2.0
# -----------------------------------------------------------------------------
# JOINT SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.joint_sensor = CN()
_C.habitat.task.joint_sensor.type = "JointSensor"
_C.habitat.task.joint_sensor.dimensionality = 7
# -----------------------------------------------------------------------------
# END EFFECTOR POSITION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.end_effector_sensor = CN()
_C.habitat.task.end_effector_sensor.type = "EEPositionSensor"
# -----------------------------------------------------------------------------
# IS HOLDING SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.is_holding_sensor = CN()
_C.habitat.task.is_holding_sensor.type = "IsHoldingSensor"
# -----------------------------------------------------------------------------
# RELATIVE RESTING POSISITON SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.relative_resting_pos_sensor = CN()
_C.habitat.task.relative_resting_pos_sensor.type = (
    "RelativeRestingPositionSensor"
)
# -----------------------------------------------------------------------------
# JOINT VELOCITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.joint_velocity_sensor = CN()
_C.habitat.task.joint_velocity_sensor.type = "JointVelocitySensor"
_C.habitat.task.joint_velocity_sensor.dimensionality = 7
# -----------------------------------------------------------------------------
# ORACLE NAVIGATION ACTION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.oracle_nav_action_SENSOR = CN()
_C.habitat.task.oracle_nav_action_SENSOR.type = "OracleNavigationActionSensor"
# -----------------------------------------------------------------------------
# RESTING POSITION SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.resting_pos_sensor = CN()
_C.habitat.task.resting_pos_sensor.type = "RestingPositionSensor"
# -----------------------------------------------------------------------------
# ART JOINT SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.ART_joint_sensor = CN()
_C.habitat.task.ART_joint_sensor.type = "ArtJointSensor"
# -----------------------------------------------------------------------------
# NAV GOAL SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.NAV_goal_sensor = CN()
_C.habitat.task.NAV_goal_sensor.type = "NavGoalSensor"
# -----------------------------------------------------------------------------
# ART JOINT NO VELOCITY SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.ART_joint_sensor_NO_VEL = CN()
_C.habitat.task.ART_joint_sensor_NO_VEL.type = "ArtJointSensorNoVel"
# -----------------------------------------------------------------------------
# MARKER RELATIVE POSISITON SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.marker_rel_pos_sensor = CN()
_C.habitat.task.marker_rel_pos_sensor.type = "MarkerRelPosSensor"
# -----------------------------------------------------------------------------
# TARGET START SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.target_start_sensor = CN()
_C.habitat.task.target_start_sensor.type = "TargetStartSensor"
_C.habitat.task.target_start_sensor.goal_format = "CARTESIAN"
_C.habitat.task.target_start_sensor.dimensionality = 3
# -----------------------------------------------------------------------------
# OBJECT SENSOR
# -----------------------------------------------------------------------------
_C.habitat.task.object_sensor = CN()
_C.habitat.task.object_sensor.type = "TargetCurrentSensor"
_C.habitat.task.object_sensor.goal_format = "CARTESIAN"
_C.habitat.task.object_sensor.dimensionality = 3
=======
>>>>>>> upstream/main

def get_full_config_path(config_path: str, configs_dir: str) -> str:
    r"""Returns absolute path to the yaml config file if exists, else raises RuntimeError.

    :param config_path: path to the yaml config file.
    :param configs_dir: path to the config files root directory.
    :return: absolute path to the yaml config file.
    """
    if osp.exists(config_path):
        return osp.abspath(config_path)

    proposed_full_path = osp.join(configs_dir, config_path)
    if osp.exists(proposed_full_path):
        return osp.abspath(proposed_full_path)

    raise RuntimeError(f"No file found for config '{config_path}'")


get_full_habitat_config_path = partial(
    get_full_config_path, configs_dir=_HABITAT_CFG_DIR
)
get_full_habitat_config_path.__doc__ = r"""
Returns absolute path to the habitat yaml config file if exists, else raises RuntimeError.

:param config_path: relative path to the habitat yaml config file.
:return: absolute config to the habitat yaml config file.
"""


def get_agent_config(
    sim_config: DictConfig, agent_id: Optional[int] = None
) -> DictConfig:
    r"""Returns agent's config node of default agent or based on index of the agent.

    :param sim_config: config of :ref:`habitat.core.simulator.Simulator`.
    :param agent_id: index of the agent config (relevant for multi-agent setup).
    :return: relevant agent's config.
    """
    if agent_id is None:
        agent_id = sim_config.default_agent_id

    agent_name = sim_config.agents_order[agent_id]
    agent_config = sim_config.agents[agent_name]

    return agent_config


lock = threading.Lock()


def get_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
    configs_dir: str = _HABITAT_CFG_DIR,
) -> DictConfig:
    r"""Returns habitat config object composed of configs from yaml file (config_path) and overrides.

    :param config_path: path to the yaml config file.
    :param overrides: list of config overrides. For example, :py:`overrides=["habitat.seed=1"]`.
    :param configs_dir: path to the config files root directory (defaults to :ref:`_HABITAT_CFG_DIR`).
    :return: composed config object.
    """
    register_hydra_plugin(HabitatConfigPlugin)

    config_path = get_full_config_path(config_path, configs_dir)
    # If get_config is called from different threads, Hydra might
    # get initialized twice leading to issues. This lock fixes it.
    with lock, initialize_config_dir(
        version_base=None,
        config_dir=osp.dirname(config_path),
    ):
        cfg = compose(
            config_name=osp.basename(config_path),
            overrides=overrides if overrides is not None else [],
        )

    # In the single-agent setup use the agent's key from `habitat.simulator.agents`.
    sim_config = cfg.habitat.simulator
    if len(sim_config.agents) == 1:
        with read_write(sim_config):
            sim_config.agents_order = list(sim_config.agents.keys())

    # Check if the `habitat.simulator.agents_order`
    # is set and matches the agents' keys in `habitat.simulator.agents`.
    assert len(sim_config.agents_order) == len(sim_config.agents) and set(
        sim_config.agents_order
    ) == set(sim_config.agents.keys()), (
        "habitat.simulator.agents_order should be set explicitly "
        "and match the agents' keys in habitat.simulator.agents.\n"
        f"habitat.simulator.agents_order: {sim_config.agents_order}\n"
        f"habitat.simulator.agents: {list(sim_config.agents.keys())}"
    )

    OmegaConf.set_readonly(cfg, True)

    return cfg
