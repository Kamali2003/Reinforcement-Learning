import random

class PacManEnvironment:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pacman_x = 1
        self.pacman_y = 1
        self.ghost_x = width - 2
        self.ghost_y = height - 2
        self.food_x = random.randint(1, width - 2)
        self.food_y = random.randint(1, height - 2)
        self.score = 0
        self.done = False

    def reset(self):
        self.pacman_x = 1
        self.pacman_y = 1
        self.ghost_x = self.width - 2
        self.ghost_y = self.height - 2
        self.food_x = random.randint(1, self.width - 2)
        self.food_y = random.randint(1, self.height - 2)
        self.score = 0
        self.done = False

    def get_state(self):
        return (self.pacman_x, self.pacman_y, self.ghost_x, self.ghost_y, self.food_x, self.food_y)

    def take_action(self, action):
        if action == 0:  # Move up
            self.pacman_y -= 1
        elif action == 1:  # Move down
            self.pacman_y += 1
        elif action == 2:  # Move left
            self.pacman_x -= 1
        elif action == 3:  # Move right
            self.pacman_x += 1

        self.pacman_x = max(1, min(self.pacman_x, self.width - 2))
        self.pacman_y = max(1, min(self.pacman_y, self.height - 2))

        if self.pacman_x == self.ghost_x and self.pacman_y == self.ghost_y:
            self.score -= 10
            self.done = True

        if self.pacman_x == self.food_x and self.pacman_y == self.food_y:
            self.score += 10
            self.food_x = random.randint(1, self.width - 2)
            self.food_y = random.randint(1, self.height - 2)

    def is_done(self):
        return self.done


class DuelingDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 0.1
        self.q_table = {}

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        return max(range(self.action_size), key=lambda x: self.q_table.get((state, x), 0))

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = max(range(self.action_size), key=lambda x: self.q_table.get((next_state, x), 0))
        old_value = self.q_table.get((state, action), 0)
        next_value = self.q_table.get((next_state, best_next_action), 0)
        new_value = (1 - 0.1) * old_value + 0.1 * (reward + next_value)
        self.q_table[(state, action)] = new_value


# Main training loop
if __name__ == "__main__":
    width = 5
    height = 5
    state_size = 6
    action_size = 4
    env = PacManEnvironment(width, height)
    agent = DuelingDQNAgent(state_size, action_size)
    episodes = 1000

    for episode in range(episodes):
        env.reset()
        total_reward = 0

        while not env.is_done():
            state = env.get_state()
            action = agent.select_action(state)
            env.take_action(action)
            next_state = env.get_state()
            reward = env.score
            agent.update_q_table(state, action, reward, next_state)
            total_reward += reward

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
