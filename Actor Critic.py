import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
seed = 42
gamma = 0.99
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")
env.seed(seed)
eps = np.finfo(np.float32).eps.item()
num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
common = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs=inputs, outputs=[action, critic])
optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:
  state = env.reset()
  episode_reward = 0
  with tf.GradientTape() as tape:
    for timestep in range(1, max_steps_per_episode):
      state = tf.convert_to_tensor(state)
      state = tf.expand_dims(state, 0)
      action_probs, critic_value = model(state)
      critic_value_history.append(critic_value[0, 0])
      action = np.random.choice(num_actions, p=np.squeeze(action_probs))
      action_probs_history.append(tf.math.log(action_probs[0, action]))
      state, reward, done, _ = env.step(action)
      rewards_history.append(reward)
      episode_reward += reward
if done:
  break
running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
returns = []
discounted_sum = 0
for r in rewards_history[::-1]:
  discounted_sum = r + gamma * discounted_sum
  returns.insert(0, discounted_sum)

returns = np.array(returns)
returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
returns = returns.tolist()
history = zip(action_probs_history, critic_value_history, returns)
actor_losses = []
critic_losses = []
for log_prob, value, ret in history:
  diff = ret - value
  actor_losses.append(-log_prob * diff) # actor loss
  critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
  loss_value = sum(actor_losses) + sum(critic_losses)
  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  action_probs_history.clear()
  critic_value_history.clear()
  rewards_history.clear()
  episode_count += 1
  if episode_count % 10 == 0:
    template = "running reward: {:.2f} at episode {}"
    print(template.format(running_reward, episode_count))
