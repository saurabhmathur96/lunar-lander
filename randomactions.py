import gym

env = gym.make('LunarLander-v2')
env.reset()
for _ in range(1000):
    env.render()
    # take a random action
    env.step(env.action_space.sample())
env.close()