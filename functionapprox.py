import numpy as np
from tqdm import trange
from collections import defaultdict
from functools import partial
import gym
import tiles3



def argmax_random(a):
    a = np.array(a)
    return np.random.choice(np.flatnonzero(a == a.max()))


class LunarLanderTileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, num_tiles=8):
        """
        Initializes the MountainCar Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the
                     tile coder are the same
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        self.iht = tiles3.IHT(iht_size)
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
    
    def get_tiles(self, observation):
        scales = [self.num_tiles / 2]*(len(observation)-2) # + [self.num_tiles]*2
        tiles = tiles3.tiles(self.iht, self.num_tilings, [oi*si for oi, si in zip(observation[:-2], scales)], observation[-2:])
        return np.array(tiles)


num_tilings = 32
num_tiles = 8
iht_size = 12288
num_actions = 4 

def show_episode(env, agent):
    action_count = env.action_space.n
    state =  env.reset() # start state

    done = False
    while not done:
        env.render()

        action = agent.select_action(state, epsilon=0)
        state, reward, done, info = env.step(action) 


class SarsaAgent:
    def __init__(self, action_count, discount, step_size):
        ''' Creates an Agent that learns an Epsilon Greedy Policy using the SARSA algorithm'''
        self.action_count = action_count

        self.tc = LunarLanderTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        self.w = np.ones((num_actions, iht_size)) * 0.1
        self.discount = discount
        self.step_size = step_size

    def select_action(self, state, epsilon=0.2):
        state = self.tc.get_tiles(state)

        action_values = [sum(q[state]) for q in self.w]
        if np.random.random() < epsilon:
            return np.random.choice(self.action_count)
        else:
            return argmax_random(action_values)



    def update_policy(self, state, action, reward, next_state, next_action):
        state = self.tc.get_tiles(state)
        next_state = self.tc.get_tiles(next_state)

        # Qlearning
        # next_action_value = max([sum(q[next_tiles]) for q in w])
        next_action_value = sum(self.w[next_action, next_state])
        delta = reward + self.discount*next_action_value - sum(self.w[action, state])

        self.w[action, state] += self.step_size*delta
    
    def update_policy_on_end(self, state, action, reward):
        state = self.tc.get_tiles(state)

        delta = reward - sum(self.w[action, state]) 
        self.w[action, state] += self.step_size*delta



class QLearningAgent:
    def __init__(self, action_count, discount, step_size):
        ''' Creates an Agent that learns an Epsilon Greedy Policy using the SARSA algorithm'''
        self.action_count = action_count

        self.tc = LunarLanderTileCoder(iht_size=iht_size, num_tilings=num_tilings, num_tiles=num_tiles)

        self.w = np.ones((num_actions, iht_size)) * 0.1
        self.discount = discount
        self.step_size = step_size

    def select_action(self, state, epsilon=0.2):
        state = self.tc.get_tiles(state)

        action_values = [sum(q[state]) for q in self.w]
        if np.random.random() < epsilon:
            return np.random.choice(self.action_count)
        else:
            return argmax_random(action_values)



    def update_policy(self, state, action, reward, next_state):
        state = self.tc.get_tiles(state)
        next_state = self.tc.get_tiles(next_state)

        # Qlearning
        next_action_value = max([sum(q[next_state]) for q in self.w])
        # next_action_value = sum(self.w[next_action, next_state])
        delta = reward + self.discount*next_action_value - sum(self.w[action, state])

        self.w[action, state] += self.step_size*delta
    
    def update_policy_on_end(self, state, action, reward):
        state = self.tc.get_tiles(state)

        delta = reward - sum(self.w[action, state]) 
        self.w[action, state] += self.step_size*delta


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    action_count = env.action_space.n

    step_size = 0.5 / num_tilings
    decay = 0.99
    discount = 0.99
    epsilon = 0.2

    # agent = SarsaAgent(action_count, discount, step_size)
    agent = QLearningAgent(action_count, discount, step_size)

    for episode in range(1000):
        state = env.reset()
        action = agent.select_action(state, epsilon)

        total_reward = 0
        done = False
        while not done:
            next_state, reward, done, info = env.step(action) 

            total_reward += reward

            if done:
                agent.update_policy_on_end(state, action, reward)
                break
            else:
                next_action = agent.select_action(next_state, epsilon)
                
                # agent.update_policy(state, action, reward, next_state, next_action)
                agent.update_policy(state, action, reward, next_state)


            state = next_state
            action = next_action
        
        if episode % 100 == 0:
            print (total_reward)
            agent.step_size *= decay
            agent.step_size = max(0.005, agent.step_size)
            epsilon *= decay
            epsilon = max(0.005, epsilon)
            show_episode(env, agent)

    env.close()