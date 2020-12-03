import gym
from gym.spaces import Box
import numpy as np 
import cv2

from math import pi, cos, sin

class ContinuousNavigation2DEnv(gym.Env):
	metadata = {'render.modes': ['human']}

	def __init__(self):
		# Map settings
		self.map_height = 300.0
		self.map_width = 400.0
		self.random_init_pos = False
		self.random_target_pos = False

		# Agent settings
		self.speed_scale = 5

		# Interfaces
		self.obs_old_actions_included = False
		self.obs_length = 4 # target_pos + agent_pos
		if self.obs_old_actions_included:
			self.obs_length += 2

		self.observation_space = Box(low=0.0, high=self.map_width, shape=[self.obs_length])

		self.action_length = 2 # speed + orientation
		self.action_space = Box(low=0., high=1., shape=[self.action_length])

		# Overall settings
		self.max_steps = 200
		self._max_episode_steps = self.max_steps
		self.target_range = 50.0

		self.reset()

	def step(self, action):
		old_states = self.states.copy()
		v = action[0] * self.speed_scale
		theta = action[1] * 2 * pi

		dx = v * cos(theta) # Zero point is east
		dy = v * sin(theta)

		self.agent_pos = np.array([np.clip(old_states[0] + dx, 0.0, self.map_width), np.clip(old_states[1] + dy, 0.0, self.map_height)])
		self.states = np.concatenate((self.agent_pos, self.target_pos))
		if self.obs_old_actions_included:
			self.states = np.concatenate((self.states, self.agent_old_actions))

		self.agent_old_actions = action

		reward = -1
		done = False
		info = {}

		if np.linalg.norm(self.agent_pos - self.target_pos) < self.target_range:
			reward = 1
			done = True

		self.steps += 1
		if self.steps >= self.max_steps:
			done = True

		return self.states, reward, done, info

	def reset(self):
		self.steps = 0

		if self.random_init_pos:
			self.agent_pos = np.random.random([2])
			self.agent_pos *= [self.map_width, self.map_height]
		else:
			self.agent_pos = np.array([200.0, 150.0])

		if self.random_target_pos:
			self.target_pos = np.random.random([2])
			self.target_pos *= [self.map_width, self.map_height]
		else:
			self.target_pos = np.array([250.0, 250.0])

		self.agent_old_actions = [0.0, 0.0]
		self.states = np.concatenate((self.agent_pos, self.target_pos))
		if self.obs_old_actions_included:
			self.states = np.concatenate((self.states, self.agent_old_actions))

		return self.states    		

	def render(self, mode='human', close=False):
		print('Current step: {0}/{1}'.format(self.steps, self.max_steps))

	def _render_trajectory(self, traj, render=False):
		scale = 2
		img = np.ones((int(self.map_height * scale), int(self.map_width * scale), 3), np.float32)

		x1 = [0, 0]
		x2 = [0, 0]
		for i in range(len(traj)):
			states, _, reward, done = traj[i]
			if i == 0:
				x1 = states[0:2]
				cv2.circle(img, tuple([int(round(x*scale)) for x in x1]), 2*scale, (0.3, 0.3, 0.3), -1) # the dot of the origin
				tx = states[2:4]
				cv2.circle(img, tuple([int(round(x*scale)) for x in tx]), int(round(self.target_range*scale)), (0.7, 0.5, 0.2), -1) # the circle of target
			else:
				x2 = states[0:2]
				cv2.line(img, tuple([int(round(x*scale)) for x in x1]), tuple([int(round(x*scale)) for x in x2]), (0.3, i/float(self.max_steps), 0.3), 2*scale)
				x1 = x2

		if render:
			cv2.imshow('Trajectory', img)
			cv2.waitKey(100)
			cv2.destroyAllWindows()

		return img	

class ContinuousNavigation2DNREnv(ContinuousNavigation2DEnv):
	def __init__(self):
		super().__init__()

	def step(self, action):
		old_states = self.states.copy()
		v = action[0] * self.speed_scale
		theta = action[1] * 2 * pi

		dx = v * cos(theta) # Zero point is east
		dy = v * sin(theta)

		self.agent_pos = np.array([np.clip(old_states[0] + dx, 0.0, self.map_width), np.clip(old_states[1] + dy, 0.0, self.map_height)])
		self.states = np.concatenate((self.agent_pos, self.target_pos))
		if self.obs_old_actions_included:
			self.states = np.concatenate((self.states, self.agent_old_actions))

		self.agent_old_actions = action

		reward = 0
		done = False
		info = {}

		self.steps += 1
		if self.steps >= self.max_steps:
			done = True

		return self.states, reward, done, info