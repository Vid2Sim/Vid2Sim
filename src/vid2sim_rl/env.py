import random
import numpy as np
import gymnasium as gym

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from gymnasium import spaces
from torchvision import transforms as T

import time
import torch
import torch.nn.functional as F
from hydra.utils import to_absolute_path


class HERWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(HERWrapper, self).__init__(env)
        obs_space = env.observation_space
        # Just a placeholder
        new_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        raw_obs_space_dict = {"{}".format(k): v for k, v in obs_space.spaces.items()}
        self.observation_space = spaces.Dict({
            'observation': new_obs_space,
            'desired_goal': goal_space,
            'achieved_goal': goal_space,
            **raw_obs_space_dict
        })

    def observation(self, observation):
        flat_obs = np.zeros([1,])
        desired_goal = np.zeros([2,])  # TODO
        achieved_goal = np.zeros([2,])  # TODO
        new_obs = {"{}".format(k): v for k, v in observation.items()}
        return {
            'observation': flat_obs,
            'desired_goal': desired_goal,
            'achieved_goal': achieved_goal,
            **new_obs
        }

    # def compute_reward(self, achieved_goal, desired_goal, info):
    #     print(111)  # TODO
    #     rewards = np.zeros((len(achieved_goal),))
    #     return rewards


class UnityEnvWrapper(gym.Env):
    def __init__(self, unity_env, inference_mode, env_cfg):
        super(UnityEnvWrapper, self).__init__()
        self.unity_env = unity_env
        self.env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=True, action_space_seed=42)
        self.inference_mode = inference_mode
        self.env_cfg = env_cfg

        vis_space = self.env.observation_space[0]
        vec_space = self.env.observation_space[-1]

        self.obs_height = self.env_cfg.obs_height
        self.obs_width = self.env_cfg.obs_width

        self.stack_num = 6 #vis_space.shape[0]
        self.channel = int(self.env_cfg.use_rgb) * 3 + int(self.env_cfg.use_depth)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(low=0, high=1, shape=(self.stack_num, self.channel, self.obs_height, self.obs_width), dtype=np.float32),
            'vector': spaces.Box(low=-np.inf, high=np.inf, shape=(vec_space.shape[0]-2*self.stack_num,), dtype=np.float32),
        })
        self.action_space = self.env.action_space

        if not isinstance(self.action_space, spaces.Box):
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self._process_observation(obs)
        info = obs.pop('info')
        return obs, info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._process_observation(obs, done)
        extra_info = obs.pop('info')
        info.update(extra_info)
        return obs, reward, done, done, info

    @torch.no_grad()
    def _process_observation(self, observation, done=False):
        depth_obs, rgb_obs, vec_obs = observation
        
        # Temp for demo
        # trd_obs = depth_obs.copy()
        # depth_obs = rgb_obs.copy()[:6] 

        rgb_obs = rgb_obs.reshape(-1, 3, *rgb_obs.shape[1:]) 
        depth_obs = depth_obs.reshape(-1, 1, *depth_obs.shape[1:])
        raw_img, raw_depth = None, None

        if self.inference_mode:
            raw_img = rgb_obs.copy()
            raw_depth = depth_obs.copy() 

        H, W = rgb_obs.shape[2:] if self.env_cfg.use_rgb else depth_obs.shape[2:]
        if H != self.obs_height or W != self.obs_width:
            if self.env_cfg.use_rgb:
                rgb_obs = F.interpolate(torch.tensor(rgb_obs), size=(self.obs_height, self.obs_width), mode='nearest')
                rgb_obs = rgb_obs.numpy()
            if self.env_cfg.use_depth:
                depth_obs = F.interpolate(torch.tensor(depth_obs), size=(self.obs_height, self.obs_width), mode='nearest')
                depth_obs = depth_obs.numpy()

        N = self.stack_num
        vec_obs = vec_obs.reshape(N, -1)
        collision_info, vec_obs = vec_obs[:,:2], vec_obs[:,2:]
        
        vec_obs = vec_obs.reshape(-1)
        collision_info = collision_info.reshape(-1)

        visual_obs = None
        if self.env_cfg.use_depth:
            visual_obs = depth_obs
        if self.env_cfg.use_rgb:
            visual_obs = np.concatenate([rgb_obs, visual_obs], axis=1) if visual_obs is not None else rgb_obs

        info = {
            'collision': collision_info,
            'raw_img': raw_img,
            'raw_depth': raw_depth,
            # 'trd_view': trd_obs,
        }

        return {
            'image': visual_obs, 
            'vector': vec_obs,
            'info': info,
        }

    def close(self):
        self.env.close()


def build_env(env_path, cfg, worker_id=0, random_seed=0, inference_mode=False, no_graphics=False):
    conf_channel = EngineConfigurationChannel()
    param_channel = EnvironmentParametersChannel()
    abs_env_path = to_absolute_path(env_path)
    unity_additional_args = getattr(cfg.env, 'unity_args', [])
    unity_env = UnityEnvironment(
        file_name=abs_env_path,
        seed=cfg.env.seed,
        side_channels=[conf_channel, param_channel],
        no_graphics=no_graphics,
        worker_id=worker_id,
        additional_args=unity_additional_args if len(unity_additional_args) > 0 else None,
    )
    conf_channel.set_configuration_parameters(
        time_scale=cfg.env.time_scale, #if not inference_mode else 1.0,
        width=cfg.env.screen_width,
        height=cfg.env.screen_height
    )
    param_channel.set_float_parameter("timePenaltyMultiplier", cfg.env.time_penalty_multiplier)
    param_channel.set_float_parameter("distanceRewardMultiplier", cfg.env.distance_reward_multiplier)
    param_channel.set_float_parameter("steerSmoothRewardMultiplier", cfg.env.steer_smooth_reward_multiplier)
    param_channel.set_float_parameter("collisionPenalty", cfg.env.collision_penalty)
    param_channel.set_float_parameter("goalReward", cfg.env.goal_reward)
    param_channel.set_float_parameter("failReward", cfg.env.fail_reward)
    param_channel.set_float_parameter("maxEpisodeDuration", cfg.env.max_episode_length)
    param_channel.set_float_parameter("enableRandomObjPlacement", float(cfg.env.random_obj_placement))
    param_channel.set_float_parameter("maxLoadedObjNum", float(cfg.env.max_loaded_obj_num))
    param_channel.set_float_parameter("collisionLimit", float(cfg.env.collision_limit))
    param_channel.set_float_parameter("enableDynamicAgent", float(cfg.env.enable_dynamic_agent))
    param_channel.set_float_parameter("dynamicAgentNumber", float(cfg.env.dynamic_agent_n))
    param_channel.set_float_parameter("randomSeed", float(random_seed))
    env = UnityEnvWrapper(unity_env, inference_mode, cfg.env)
    if hasattr(cfg.train, 'HER'):
        env = HERWrapper(env)
    return env

def make_env(env_path, cfg, worker_id=None, random_seed=0, no_graphics=False, inference=False):
    if worker_id is None:
        worker_id = random.randint(0, 65535)  

    def _init():
        try:
            env = build_env(env_path, cfg, worker_id, random_seed=random_seed, inference_mode=inference, no_graphics=no_graphics)
        except Exception as e:
            print(f"Error initializing environment: {e}")
            env = None
        return env
    return _init

def auto_build_env(env_paths, cfg, work_ids=None, build_fn=make_env):
    envs = []
    if work_ids is None:
        work_ids = []
    idx = 0
    while True:
        work_id = 256*cfg.env.seed + random.randint(1, 256)
        if work_id in work_ids or work_id > 65535:
            continue
        work_ids.append(work_id)
        env = build_fn(env_paths[idx], cfg, work_id, random_seed=idx) # Different random seed for each env
        envs.append(env)
        work_ids.append(work_id)
        idx += 1
        if len(envs) == len(env_paths):
            break
    return envs, work_ids