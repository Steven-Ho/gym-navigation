from gym.envs.registration import register

register(
	id='2d-navigation-v0',
	entry_point='gym_navigation.envs:ContinuousNavigation2DEnv',
)
