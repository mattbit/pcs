"""
Deep Q-learning for the classical cart-pole problem.
====================================================

The cart-pole problem consists in a cart with a pole anchored on top. At each
step a state s is observed:

    s = [x, x_dot, theta, theta_dot]

State components are continuous. Possible actions are left (0) and right (1).
Here we use a neural net to approximate the action value function Q.
"""

import gym
import torch
import random
import numpy as np
import collections
from torch.autograd import Variable

env = gym.make('CartPole-v1')

class Memory(object):
    def __init__(self, size):
        """Fixed size memory.

        Memorize a fixed number of objects, replacing the older with the newer.
        """
        self.size = size
        self.buffer = collections.deque(maxlen=size)

    def push(self, obj):
        """Add an object to the memory."""
        self.buffer.append(obj)

    def sample(self, size):
        """Sample a batch of objects from the memory."""
        return random.sample(self.buffer, size)

    def __len__(self):
        return len(self.buffer)

# Set up the neural network
DQNetwork = torch.nn.Sequential(
    torch.nn.Linear(4, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 2))

loss_fn = torch.nn.SmoothL1Loss()  # Huber loss
optimizer = torch.optim.Adam(DQNetwork.parameters(), lr=1e-3)

# Set up the policy

def policy(s, a, DQNetwork, eps):
    """Returns the probability of choosing action `a` from state `s`."""
    # Qs contains the expected Q for all possible actions.
    Qs = DQNetwork(Variable(torch.Tensor(state))).data.numpy()

    # Softmax function (with some tricks to avoid overflow).
    return np.exp((Qs[a] - max(Qs))/eps) / np.sum(np.exp((Qs - max(Qs))/eps))

def choose_action(state, DQNetwork, eps=0.1):
    """Choose an action with the probabilities given by the policy."""
    actions = [0, 1]  # possible actions

    return int(np.random.choice(actions, p=policy(state, actions, DQNetwork, eps)))

# Train the model
gamma = 0.8
batch_size = 50
mem = Memory(1000)
for episode in range(500):
    state = env.reset()
    for step in range(500):
        # Choose the action based on the current policy (note that it uses a
        # a softmax where epsilon is the exploration parameter).
        epsilon = max(0.05, 0.9 * np.exp(-episode/200))
        action = choose_action(state, DQNetwork, eps=epsilon)
        next_state, reward, done, _ = env.step(action)

        # Add a negative reward when failing.
        if done:
            reward = -1.

        # Save current observation to memory.
        mem.push((state, action, reward, next_state))

        # If the memory contains enough data, train the model.
        if len(mem) >= batch_size:
            # We process the data in batches.
            batch = mem.sample(batch_size)

            # Split the batch columns and create the torch Variables.
            state_b, action_b, reward_b, next_state_b = zip(*batch)
            states = torch.cat([torch.Tensor([s]) for s in state_b])
            actions = torch.cat([torch.LongTensor([[a]]) for a in action_b])
            rewards = torch.cat(torch.Tensor([b]) for b in reward_b)
            next_states = torch.cat(torch.Tensor([s]) for s in next_state_b)
            state_batch = Variable(states)
            action_batch = Variable(actions)
            reward_batch = Variable(rewards)
            next_state_batch = Variable(next_states)

            # Our old estimate.
            estimate = DQNetwork(state_batch).gather(1, action_batch)

            # Our new target based on the measured reward.
            max_DQN_next, _ = DQNetwork(next_state_batch).detach().max(1)
            target = reward_batch + gamma * max_DQN_next

            # Calculate the loss function between the estimate and the target.
            # We get a single value (the mean loss of the whole batch).
            loss = loss_fn(estimate, target)
            if done:
                print("Episode: {:4}\tloss: {}".format(episode, loss.data[0]))

            # Backpropagate errors and adjust the model parameters.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Show must go on, unless we failed.
        state = next_state

        if done:
            break


print("Training finished!")

# Testing the trained model.

state = env.reset()
tot_reward = 0
for t in range(500):
    action = choose_action(state, DQNetwork, eps=0.000001)
    state, reward, done, _ = env.step(action)

    tot_reward += reward

    env.render()

    if done:
        break

env.close()
print("Test finished. Total reward: {}".format(tot_reward))
