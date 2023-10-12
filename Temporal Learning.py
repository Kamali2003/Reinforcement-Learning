import random

class RandomWalkEnvironment:
    def __init__(self, num_states):
        self.num_states = num_states
        self.current_state = num_states // 2  # Start in the middle
        self.actions = [-1, 1]  # Left and right

    def reset(self):
        self.current_state = self.num_states // 2

    def step(self, action):
        reward = 0
        done = False

        new_state = self.current_state + action
        if new_state == 0:
            reward = -1
            done = True
        elif new_state == self.num_states - 1:
            reward = 1
            done = True

        self.current_state = new_state
        return new_state, reward, done

    def render(self):
        env_str = "|" + " " * self.num_states + "|"
        env_str = env_str[:self.current_state+1] + "X" + env_str[self.current_state+2:]
        print(env_str)
class TDLearner:
    def __init__(self, num_states, alpha, gamma):
        self.num_states = num_states
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.values = [0] * num_states

    def update(self, state, reward, next_state):
        td_target = reward + self.gamma * self.values[next_state]
        td_error = td_target - self.values[state]
        self.values[state] += self.alpha * td_error

    def get_action(self, state):
        # Epsilon-greedy policy
        if random.random() < 0.5:
            return random.choice([-1, 1])
        else:
            if state == 0:
                return 1
            elif state == self.num_states - 1:
                return -1
            else:
                return -1 if self.values[state - 1] > self.values[state + 1] else 1
def main():
    num_states = 7
    env = RandomWalkEnvironment(num_states)
    td_learner = TDLearner(num_states, alpha=0.1, gamma=1.0)

    num_episodes = 10
    for episode in range(num_episodes):
        state = env.current_state
        env.reset()
        done = False

        print(f"Episode {episode + 1}")
        env.render()

        while not done:
            action = td_learner.get_action(state)
            next_state, reward, done = env.step(action)
            td_learner.update(state, reward, next_state)
            state = next_state
            env.render()

    print("Learned values:", td_learner.values)

if __name__ == "__main__":
    main()
