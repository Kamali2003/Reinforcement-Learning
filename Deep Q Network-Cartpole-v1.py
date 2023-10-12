import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the Q-network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(24, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(2, activation='linear')  # 2 output nodes for left and right actions
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training parameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
target_update_frequency = 100
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())

# Function to select an action based on epsilon-greedy policy
def select_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()  # Explore by taking a random action
    else:
        q_values = model.predict(state.reshape(1, -1))
        return np.argmax(q_values)

# Function to update the target Q-network
def update_target_model():
    target_model.set_weights(model.get_weights())

# Training loop
episodes = 10
rewards = []

# Lists to store episode rewards and their moving average
episode_rewards = []
moving_avg_rewards = []

# Visualization: Initialize a figure for plotting rewards
plt.figure(figsize=(10, 5))
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('CartPole DQN Training Progress')

for episode in range(episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)
        next_state, reward, done, _ = env.step(action)

        target = model.predict(state.reshape(1, -1))
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + gamma * np.max(target_model.predict(next_state.reshape(1, -1)))

        with tf.GradientTape() as tape:
            q_values = model(state.reshape(1, -1))
            loss = loss_fn(target, q_values)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        state = next_state
        total_reward += reward

        if done:
            rewards.append(total_reward)
            episode_rewards.append(total_reward)

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            if episode % target_update_frequency == 0:
                update_target_model()

            if episode % 10 == 0:
                print(f"Episode {episode}, Average Reward: {np.mean(rewards[-10:])}")

            # Visualization: Plot the rewards over episodes every 50 episodes
            if episode % 50 == 0 and episode > 0:
                moving_avg = np.mean(episode_rewards[-50:])
                moving_avg_rewards.append(moving_avg)
                plt.plot(rewards, label='Episode Reward')
                plt.plot(np.arange(len(moving_avg_rewards)) * 50, moving_avg_rewards, label='Moving Average (50 episodes)')
                plt.legend()
                plt.pause(0.1)

            break

# Final visualization: Plot the rewards and moving average
moving_avg_rewards = [np.mean(episode_rewards[i-50:i+1]) if i >= 50 else np.mean(episode_rewards[:i+1]) for i in range(len(episode_rewards))]
plt.plot(rewards, label='Episode Reward')
plt.plot(np.arange(len(moving_avg_rewards)) * 50, moving_avg_rewards, label='Moving Average (50 episodes)')
plt.legend()
plt.show()

# Evaluate the trained model
total_rewards = []
for _ in range(10):
    state = env.reset()
    episode_reward = 0
    while True:
        action = np.argmax(model.predict(state.reshape(1, -1)))
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state
        if done:
            total_rewards.append(episode_reward)
            break

print("Average Reward (Evaluation):", np.mean(total_rewards))
