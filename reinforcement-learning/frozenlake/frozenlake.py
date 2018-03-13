"""
FrozenLake with simple Q-learning.
"""

import numpy as np
import gym

EPISODES = 3000
GAMMA = 0.9

env = gym.make("FrozenLake-v0")
Q = np.zeros((env.observation_space.n, env.action_space.n))

def policy(s, a, Q, eps=0.1):
    return np.exp(Q[s, a]/eps) / np.sum(np.exp(Q[s, :]/eps))


def softmax(Q, s, eps=0.1):
    return eps * np.log(np.sum(np.exp(Q[s, :]/eps)))

def alpha(n):
    """Learning rate"""
    return 0.1  # learn slowly

actions = range(env.action_space.n)


# %
for _ in range(EPISODES):
    nn = np.zeros_like(Q)
    s = env.reset()
    done = False
    for _ in range(100):
        a = np.random.choice(actions, p=policy(s, actions, Q))
        s_new, r, done, _ = env.step(a)

        if done and r == 0.0:
            # I really want to avoid the risky steps!
            r = -10.

        Q[s, a] = Q[s, a] + 0.1*(r + GAMMA*softmax(Q, s_new) - Q[s, a])
        s = s_new

        if done:
            break


Q

# %
tot_reward = 0.
for _ in range(1000):
    s = env.reset()
    done = False
    while not done:
        a = np.argmax(Q[s, :])
        s, r, done, _ = env.step(a)
        tot_reward += r

tot_reward
