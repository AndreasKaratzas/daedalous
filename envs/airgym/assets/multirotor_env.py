import cv2
import math
import time
import numpy
import airsim

from gym import spaces
from numpy import linalg as LA

from airgym.assets.airsim_env import AirSimEnv


class Multirotor(AirSimEnv):
    """OpenAI GYM environment class for AirSim Multirotor.

    ...

    Attributes
    ----------
    quant_areas : int
        Number of submatrices to split the preprocessed depth map into.
    step_length : int
        Meters of drone movement in a single action.
    """

    def __init__(self, quant_areas, step_length):
        self.drone = airsim.MultirotorClient()
        self.drone.confirmConnection()
        self.drone.simRunConsoleCommand("t.MaxFPS 10")
        self.action_space = spaces.Discrete(6)
        self.image_request = airsim.ImageRequest("0", 3, False, False)
        self.step_length = step_length
        self.quantum_size = quant_areas

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

        self.misc_shape = (
            self.state["prev_position"].shape[0] +
            self.state["curr_position"].shape[0] +
            self.state["target_coordinates"].shape[0] +
            self.state["life"].shape[0]
        )

        self.observation_space = spaces.Box(0, 255, shape=(self.quantum_size +
                                                           self.misc_shape, 1), dtype=numpy.uint8)
        self.step_thr = 0
        self.targets = numpy.array([[150, -160, - 2], [105, 215, -28],
                                    [-140, -126, - 2], [- 65, 110, -
                                                        40], [160, - 90, - 6], [150, 31, -15],
                                    [120, - 84, - 6], [54, - 15, -
                                                       17], [125, - 15, -21], [10, - 95, - 7],
                                    [- 40, -150, -50], [95, 125, -
                                                        40], [75, 5, -12], [125, 35, -11],
                                    [- 50, 60, - 6], [- 40, 130, -10]], dtype=numpy.float)
        self.clockspeed = 1000
        self.est_thr = 3.0

    def __del__(self):
        self.drone.reset()
        self.drone.armDisarm(False)
        self.drone.enableApiControl(False)

    def _setup_flight(self, target=None):
        """Sets up drone in flight mode.

        Parameters
        ----------
        target : numpy.ndarray, optional
            Specific target coordinates, by default None.
        """
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.drone.moveToZAsync(z=-20, velocity=10, timeout_sec=10).join()

        self.state["landed"] = False
        self.state["arrived"] = False
        self.state["collision"] = False
        self.state["target_coordinates"] = self.targets[
            numpy.random.choice(self.targets.shape[0])] \
            if type(target) == type(None) else target
        self.state["curr_position"] = numpy.array([0, 0, 0], dtype=numpy.float)
        self.state["step_counter"] = 0
        self.state["life"] = numpy.array([1], dtype=numpy.float)

        self.step_thr = int(numpy.sum(numpy.ceil(
            numpy.abs(self.state["target_coordinates"]) /
            self.step_length))) * 3

    def _transform_obs(self, obs):
        """Typecasts raw AirSim ImageRequest response into numpy.ndarray.

        Parameters
        ----------
        obs : airsim.types.ImageRequest Observation image of drone.

        Returns
        -------
        numpy.ndarray
            Observation image of drone.
        """
        observation = numpy.frombuffer(
            obs[0].image_data_uint8, dtype=numpy.uint8)
        observation = observation.reshape(obs[0].height, obs[0].width, 3)
        return observation

    def _permute_orientation(self, observation):
        """Transposes z-axis of the input matrix and then
        it normalizes the result.

        Parameters
        ----------
        observation : numpy.ndarray
            Grayscaled image observation of the drone.

        Returns
        -------
        numpy.ndarray
            Normalized transposed input.
        """
        observation = observation.reshape(1, 84, 84)
        observation = observation / 255.0
        return observation

    def _grayscale(self, observation):
        """Grayscale transformation of input RGB image.

        Parameters
        ----------
        observation : numpy.ndarray
            RGB image observation of drone.

        Returns
        -------
        numpy.ndarray
            Grayscale image observation of drone.
        """
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = numpy.array(observation, dtype=numpy.uint8)
        return observation

    def _resize(self, observation, shape):
        """Rescales RGB image.

        Parameters
        ----------
        observation : numpy.ndarray
            Input RGB image.
        shape : Tuple
            Target shape.

        Returns
        -------
        numpy.ndarray
            Rescaled RGB image.
        """
        shape = (shape, shape)
        observation = cv2.resize(
            observation, shape, interpolation=cv2.INTER_AREA)
        return observation

    def _split(self, matrix, nrows, ncols):
        """Splits a matrix into submatrices.

        Parameters
        ----------
        matrix : numpy.ndarray
            Input matrix.
        nrows : int
            Number of block rows.
        ncols : numpy.ndarray
            Number of block columns.

        Returns
        -------
        numpy.ndarray
            A block matrix.
        """
        _, r, h = matrix.shape
        return matrix.reshape(h // nrows, nrows, -1, ncols)\
            .swapaxes(1, 2).reshape(-1, nrows, ncols)

    def _quantization(self, observation):
        """Downscales drone's observation using mean-pooling.

        Parameters
        ----------
        observation : numpy.ndarray
            The drone's image observation.

        Returns
        -------
        numpy.ndarray
            Downscaled image observation.
        """
        c, h, w = observation.shape
        shrink_factor = (c * h * w) // self.quantum_size
        dim = int(math.sqrt(shrink_factor))
        observation = self._split(observation, dim, dim)
        observation = numpy.mean(observation, axis=(1, 2))
        return observation

    def _get_depth(self):
        """Requests a depth map from AirSim and
        preprocesses it to feed it to an RL agent.

        Returns
        -------
        numpy.ndarray
            Drone's image observation.
        """
        obs = self.drone.simGetImages([self.image_request])
        obs_img = self._transform_obs(obs)

        while obs_img.shape != (144, 256, 3):
            obs = self.drone.simGetImages([self.image_request])
            obs_img = self._transform_obs(obs)

        obs_img = self._grayscale(obs_img)
        obs_img = self._resize(obs_img, 84)
        obs_img = self._permute_orientation(obs_img)
        obs_img = self._quantization(obs_img)

        return obs_img

    def _get_misc(self):
        """Generates vector observation of drone. It merges
        current, past and target coordinates of drone including
        a life indicator.

        Returns
        -------
        numpy.ndarray
            Vector observation of drone.
        """
        drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["curr_position"]
        self.state["curr_position"] = drone_state.kinematics_estimated.position\
            .to_numpy_array()
        self.state["prev_dist"] = self.state["curr_dist"]
        self.state["curr_dist"] = numpy.sqrt(numpy.abs(
            self.state["target_coordinates"] - self.state["curr_position"]))
        self.state["Dg"] = numpy.abs(
            self.state["geo_limits"]) - numpy.abs(self.state["curr_position"])
        self.state["collision"] = self.drone.simGetCollisionInfo().has_collided
        self.state["arrived"] = LA.norm(
            self.state["curr_dist"], 2) < self.est_thr
        self.state["landed"] = (
            self.state["prev_position"] == self.state["curr_position"]).all()
        self.state["life"] = numpy.array(
            [(self.step_thr - self.state["step_counter"]) / self.step_thr], dtype=numpy.float)

        obs_misc = numpy.concatenate((
            self.state["prev_position"] / numpy.abs(self.state["geo_limits"]),
            self.state["curr_position"] / numpy.abs(self.state["geo_limits"]),
            self.state["target_coordinates"] /
            numpy.abs(self.state["geo_limits"]),
            self.state["life"]
        ))

        return obs_misc

    def _get_obs(self, curr_obs, state_misc):
        """Merges image and vector observations
        into one manipulatable vector.

        Parameters
        ----------
        curr_obs : numpy.ndarray
            Image observation.
        state_misc : numpy.ndarray
            Vector observation.

        Returns
        -------
        numpy.ndarray
            Agent's observation for a time step `t`. 
        """
        obs_img = curr_obs.flatten()
        obs_misc = state_misc.flatten()
        obs = numpy.concatenate((obs_img, obs_misc))
        return obs

    def _do_action(self, action):
        """Executes the action selected by the agent.

        Parameters
        ----------
        action : int
            An action selected by the agent.
        """
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
        """Computes the reward propagated to the agent.

        Returns
        -------
        float
            The reward of the agent at a time step `t`.
        """
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
            reward = -10
        else:
            self.state["step_counter"] += 1

        if self.state["landed"] and not done:
            reward = -10
            done = 1

        return reward, done

    def get_stats(self, reward):
        """Creates a dictionary with some stats
        of a time step `t` to print.

        Parameters
        ----------
        reward : float
            The reward of the agent at that time step.

        Returns
        -------
        Dict
            Stats of the agent at a time step `t`.
        """
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
        """Performs a time step using the
        selected action by the agent.

        Parameters
        ----------
        action : int
            The selected action by the agent.

        Returns
        -------
        Tuple
            The State, Reward, Terminal and Stats
            corresponding to that time step.
        """
        self._do_action(action)
        curr_obs = self._get_depth()
        misc = self._get_misc()
        obs = self._get_obs(curr_obs, misc)
        reward, done = self._compute_reward()
        return obs, reward, done, self.get_stats(reward)

    def reset(self, target=None):
        """Resets agent state for a new episode.

        Parameters
        ----------
        target : numpy.ndarray, optional
            Specific target coordinates, by default None.

        Returns
        -------
        numpy.ndarray
            The agent's first observation for that episode.
        """
        self._setup_flight(target)
        curr_obs = self._get_depth()
        misc = self._get_misc()
        obs = self._get_obs(curr_obs, misc)
        return obs

    def interpret_action(self, action):
        """Converts integer to action tuple.

        Parameters
        ----------
        action : int
            The action index.

        Returns
        -------
        Tuple
            The action to be executed.
        """
        if action == 0:
            # Drone forwards
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            # Drone right
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            # Drone down
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            # Drone backwards
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            # Drone left
            quad_offset = (0, -self.step_length, 0)
        else:
            # Drone up
            quad_offset = (0, 0, -self.step_length)

        return quad_offset
