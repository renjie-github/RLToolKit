import random
from collections import deque
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    def __init__(self, maxsize=100000):
        self.queue = deque(maxlen=maxsize)

    def push(self, state, action, reward, next_state, done):
        self.queue.append((state, action, reward, next_state, done))

    def get(self, batchsize):
        size = len(self.queue)
        assert size >= batchsize

        samples = random.sample(self.queue, batchsize)
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*samples))

        rewards = rewards.reshape((-1, 1))
        dones = dones.reshape((-1, 1))
        if len(actions.shape) == 1:
            actions = actions.reshape((-1, 1))
        
        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        next_states = tf.cast(next_states, dtype=tf.float32)
        dones = tf.cast(dones, dtype=tf.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.queue)

class PriorityReplayBuffer:
    def __init__(self, maxsize=100000, total_episodes=100000, alpha=0.6, beta=0.4):
        """
        prioritization component alpha, beta for compensating the non-uniform probabilities
        alpha = 0.6, beta = 0.4 for the proportional variant
        alpha = 0.7, beta = 0.5 for the rank-based variant

        reducing alpha and/or increasing beta to revert to baseline behavior
        """
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.capacity = maxsize
        self.total_episodes = total_episodes
        self.queue = deque(maxlen=maxsize)
        self.priorities = np.zeros((maxsize), dtype=np.float32)
        self.pos = 0
    
    def aneal_beta(self, epi):
        # episode starts from 1
        self.beta = self.beta_start + epi * (1. - self.beta_start) / self.total_episodes
    
    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.queue else 1.
        if len(self.queue) < self.capacity:
            self.queue.append((state, action, reward, next_state, done))
        else:
            self.queue[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def get(self, batchsize):
        size = len(self.queue)
        assert size >= batchsize

        prios = self.priorities[:size]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(size, batchsize, p=probs)
        samples = [self.queue[idx] for idx in indices]
        states, actions, rewards, next_states, dones = map(np.asarray, zip(*samples))

        weights = (size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = weights.reshape((-1, 1)).astype(np.float32)

        rewards = rewards.reshape((-1, 1))
        dones = dones.reshape((-1, 1))
        if len(actions.shape) == 1:
            actions = actions.reshape((-1, 1))

        states = tf.cast(states, dtype=tf.float32)
        actions = tf.cast(actions, dtype=tf.float32)
        rewards = tf.cast(rewards, dtype=tf.float32)
        next_states = tf.cast(next_states, dtype=tf.float32)
        dones = tf.cast(dones, dtype=tf.float32)
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def __len__(self):
        return len(self.queue)