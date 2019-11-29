from gym_navigation.envs.navigation import ContinuousNavigation2DEnv
import numpy as np
import gym
import cv2

env = gym.make('2d-navigation-v0')

# One episode
done = False
obs = env.reset()
traj = []
traj.append([obs, None, 0.0, False])
while(not done):
	action = np.random.random(2)
	obs, r, done, _ = env.step(action)
	traj.append([obs, action, r, done])
	

# print(traj)
img = env._render_trajectory(traj)
print(img)
cv2.imwrite("img.jpg", img*255.0)