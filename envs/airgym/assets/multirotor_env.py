
import os
import gym
import cv2
import math
import time
import numpy
import setup_path
import airsim

from gym import spaces
from numpy import linalg as LA

from src.core import *
from airgym.assets.airsim_env import AirSimEnv

class Multirotor(AirSimEnv):

    def __init__(self, step_length):
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        self.drone.simRunConsoleCommand("t.MaxFPS 10")
        self.action_space = spaces.Discrete(6)
        self.image_request = airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)
        self.step_length = step_length

        self.decay = 1.0
        self.counter = 0

        self.state = {
            "curr_position": numpy.array([0, 0, 0], dtype=numpy.float),
            "prev_position": numpy.array([0, 0, 0], dtype=numpy.float),
            "target_coordinates": numpy.array([0, 0, 0], dtype=numpy.float),
            "geo_limits": numpy.array([250, 250, 100], dtype=numpy.float),
            "Dg": numpy.array([0, 0, 0], dtype=numpy.float),
            "prev_dist": numpy.array([0, 0, 0], dtype=numpy.float),
            "curr_dist": numpy.array([0, 0, 0], dtype=numpy.float),
            "arrived": False,
            "collision": False,
            "landed": True,
            "step_counter": 0,
            "life": numpy.array([1], dtype=numpy.float),
        }

        self.misc_shape = self.state["curr_position"].shape[0] + self.state["target_coordinates"].shape[0]

        self.observation_space = spaces.Box(0, 255, shape=((3 * 84 * 84) + 6, 1), dtype=numpy.uint8)
        self.step_thr = 0
        self.targets = CITY_ENV_TARGETS # AIRSIM_NH_ENV_TARGETS
        self.clockspeed = CLOCKSPEED
        self.est_thr = TARGET_DISTANCE_THRESHOLD

    def __del__(self):
        self.drone.reset()
        self.drone.armDisarm(False)
        self.drone.enableApiControl(False)
    
    def _setup_flight(self, target=None):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToZAsync(z=-20, velocity=10, timeout_sec=10).join()
        
        self.decay = 1.0
        self.counter = 0

        self.state["landed"] = False
        self.state["arrived"] = False
        self.state["collision"] = False
        self.state["target_coordinates"] = self.targets[numpy.random.choice(self.targets.shape[0])] if type(target) == type(None) else target
        self.state["curr_position"] = numpy.array([0, 0, 0], dtype=numpy.float)
        self.state["life"] = numpy.array([1], dtype=numpy.float)
        self.state["step_counter"] = 0
        
        self.step_thr = int(numpy.sum(numpy.ceil(numpy.abs(self.state["target_coordinates"]) / self.step_length))) * 3

    def _transform_obs(self, obs):
        observation = numpy.frombuffer(obs[0].image_data_uint8, dtype=numpy.uint8)
        observation = observation.reshape(obs[0].height, obs[0].width, 3)
        return observation
    
    def _permute_orientation(self, observation):
        observation = observation.reshape(84, 84)
        observation = observation / 255.0
        return observation
    
    def _grayscale(self, observation):
        # uncompressed RGB array bytes
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = numpy.array(observation, dtype=numpy.uint8)
        return observation

    def _resize(self, observation, shape):
        shape = (shape, shape)
        observation = cv2.resize(observation, shape, interpolation=cv2.INTER_AREA)
        return observation
    
    def _get_drone_feed(self):
        obs = self.drone.simGetImages([self.image_request])
        obs_img = self._transform_obs(obs)
        
        while obs_img.shape != (144, 256, 3):
            obs = self.drone.simGetImages([self.image_request])
            obs_img = self._transform_obs(obs)

        obs_img = self._grayscale(obs_img)
        obs_img = self._resize(obs_img, 84)
        obs_img = self._permute_orientation(obs_img)
        
        return obs_img
    
    def _get_misc(self):
        drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["curr_position"]
        self.state["curr_position"] = drone_state.kinematics_estimated.position.to_numpy_array()
        self.state["prev_dist"] = self.state["curr_dist"]
        self.state["curr_dist"] = numpy.sqrt(numpy.abs(self.state["target_coordinates"] - self.state["curr_position"]))
        self.state["Dg"] = numpy.abs(self.state["geo_limits"]) - numpy.abs(self.state["curr_position"])
        self.state["collision"] = self.drone.simGetCollisionInfo().has_collided
        self.state["arrived"] = LA.norm(self.state["curr_dist"], 2) < self.est_thr
        self.state["landed"] = (self.state["prev_position"] == self.state["curr_position"]).all()
        self.state["life"] = numpy.array([(self.step_thr - self.state["step_counter"]) / self.step_thr], dtype=numpy.float)
        
        obs_misc = numpy.concatenate((
            self.state["curr_position"],
            self.state["target_coordinates"]
        ))

        return obs_misc

    def _get_obs(self, frame_1, frame_2, frame_3, state_misc):
        features = numpy.stack((frame_1, frame_2, frame_3), axis=0)
        obs = {'features': features, 'stats': state_misc}
        return obs
    
    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        self.drone.moveByVelocityAsync(
            quad_offset[0],
            quad_offset[1],
            quad_offset[2],
            2,
            drivetrain=airsim.DrivetrainType.ForwardOnly,
            yaw_mode=airsim.YawMode(False, 0)
        ).join()
        time.sleep(2 / self.clockspeed)
        self.drone.moveByVelocityAsync(0, 0, 0, 1).join()
        self.drone.hoverAsync().join()

    def _compute_reward(self):
        done = 0

        if self.state["collision"]:
            reward = -100
            done = 1
        elif self.state["arrived"]:
            reward = 100
            done = 1
        else:
            rel_dist = self.state["prev_dist"] - self.state["curr_dist"]
            rel_idx = numpy.abs(rel_dist).argmax(axis=0)
            reward = rel_dist[rel_idx] * 2
        
        if self.step_thr - self.state["step_counter"] < 0 and not done:
            done = 1
            reward = 0
        else:
            self.state["step_counter"] += 1

        if self.state["landed"] and not done:
            reward = -10
            done = 1
        
        if (self.state["Dg"] < 0).any():
            reward = 0
            done = 1
        
        return reward, done

    def get_stats(self, reward):
        return dict(
            Previous_position=self.state["prev_position"],
            Current_position=self.state["curr_position"],
            Target_point=self.state["target_coordinates"],
            Geofence_rel_diff=self.state["Dg"],
            Distance=self.state["curr_dist"],
            Life=self.state["life"],
            Reward=reward,
            Collided=self.state["collision"],
            Landed=self.state["landed"],
            Target_reached=self.state["arrived"]
        )
    
    def step(self, action):
        self._do_action(action)
        obs_1 = self._get_drone_feed()
        self._do_action(action)
        obs_2 = self._get_drone_feed()
        self._do_action(action)
        obs_3 = self._get_drone_feed()
        misc = self._get_misc()
        obs = self._get_obs(obs_1, obs_2, obs_3, misc)
        reward, done = self._compute_reward()
        return obs, reward, done, self.get_stats(reward)

    def reset(self, target=None):
        self._setup_flight(target)
        obs_1 = self._get_drone_feed()
        obs_2 = self._get_drone_feed()
        obs_3 = self._get_drone_feed()
        misc = self._get_misc()
        obs = self._get_obs(obs_1, obs_2, obs_3, misc)
        return obs
    
    def interpret_action(self, action):
        if action == 0:
            # forward
            quad_offset = (-self.step_length * self.decay, 0, 0)
        elif action == 1:
            # right
            quad_offset = (0, -self.step_length * self.decay, 0)
        elif action == 2:
            # down
            quad_offset = (0, 0, self.step_length * self.decay)
        elif action == 3:
            # backwards
            quad_offset = (self.step_length * self.decay, 0, 0)
        elif action == 4:
            # left
            quad_offset = (0, self.step_length * self.decay, 0)
        else:
            # up
            quad_offset = (0, 0, -self.step_length * self.decay)
        
        self.counter = self.counter + 1

        if self.counter % 3 == 0:
            self.decay = 1
        else:
            self.decay = self.decay * (1 / ((self.counter % 3) + 1))

        return quad_offset
