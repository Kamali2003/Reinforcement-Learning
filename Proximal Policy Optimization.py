import gym
env = gym.make("Taxi-v3")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Define your policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(num_actions, activation='softmax')

    def call(self, state):
        x = self.dense1(state)
        return self.dense2(x)

# Initialize policy network
num_actions = env.action_space.n
policy_network = PolicyNetwork(num_actions)

optimizer = Adam(learning_rate=0.001)
def ppo_loss(old_probs, new_probs, advantages, epsilon=0.2):
    ratio = new_probs / (old_probs + 1e-5)
    surrogate1 = ratio * advantages
    surrogate2 = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon) * advantages
    return -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

def train_ppo(policy_network, optimizer, states, actions, old_probs, advantages):
    with tf.GradientTape() as tape:
        new_probs = policy_network(states)
        loss = ppo_loss(old_probs, new_probs, advantages)
    grads = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
    # Training loop
num_episodes = 1000
max_steps = 200
discount_factor = 0.99
observation = env.reset()
tf.debugging.enable_check_numerics()
for episode in range(num_episodes):
    states = []
    actions = []
    rewards = []
    old_probs = []

    for step in range(max_steps):
        state = np.reshape(observation, [1, -1])
        states.append(state)

        action_probs = policy_network(state)
        action = np.random.choice(num_actions, p=np.squeeze(action_probs))
        actions.append(action)
        old_probs.append(action_probs[0][action])

        observation, reward, done, _ = env.step(action)
        rewards.append(reward)

        if done:
            break

    discounted_rewards = []
    advantage = 0
    for r in rewards[::-1]:
        advantage = r + discount_factor * advantage
        discounted_rewards.append(advantage)
    discounted_rewards.reverse()
    advantages = discounted_rewards

    states = tf.concat(states, axis=0)
    actions = np.array(actions)
    old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
    advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)

    train_ppo(policy_network, optimizer, states, actions, old_probs, advantages)

    if episode % 10 == 0:
        print(f"Episode: {episode}, Total Reward: {np.sum(rewards)}")
total_rewards = []

for episode in range(100):
    observation = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        state = np.reshape(observation, [1, -1])
        action_probs = policy_network(state)
        action = np.argmax(np.squeeze(action_probs))
        observation, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            break

    total_rewards.append(episode_reward)

print("Average reward over 100 episodes:", np.mean(total_rewards))
