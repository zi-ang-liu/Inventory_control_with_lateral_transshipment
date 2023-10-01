from gymnasium.envs.registration import register

register(
     id="gym_examples/LateralTransEnv-v0",
     entry_point="gym_examples.envs:LateralTransEnv",
     max_episode_steps=200,
)