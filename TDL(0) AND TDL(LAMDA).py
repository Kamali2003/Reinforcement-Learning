import numpy as np
import matplotlib.pyplot as plt

NUM_STATES = 21
START = 9
END_0 = 0
END_1 = 20
class ValueFunctionTD:
    def __init__(self, alpha=0.1, gamma=0.9, lmbda=0.8):
        self.weights = np.zeros(NUM_STATES)
        self.z = np.zeros(NUM_STATES)
        self.alpha = alpha
        self.gamma = gamma
        self.lmbda = lmbda

    def value(self, state):
        v = self.weights[state]
        return v

    def updateZ(self, state):
        dev = 1
        self.z *= self.gamma * self.lmbda
        self.z[state] += dev

    def learn(self, state, nxtState, reward):
        delta = reward + self.gamma * self.value(nxtState) - self.value(state)
        delta *= self.alpha
        self.weights += delta * self.z
class RWTD:
    def __init__(self, start=START, end=False, debug=False):
        self.actions = ["left", "right"]
        self.state = start
        self.end = end
        self.reward = 0
        self.debug = debug

    def chooseAction(self):
        action = np.random.choice(self.actions)
        return action

    def takeAction(self, action):
        new_state = self.state
        if not self.end:
            if action == "left":
                new_state = self.state - 1
            else:
                new_state = self.state + 1
            if new_state in [END_0, END_1]:
                self.end = True
        return new_state

    def giveReward(self, state):
        if state == END_0:
            return -1
        if state == END_1:
            return 1
        return 0

    def reset(self):
        self.state = START
        self.end = False
        self.states = []

    def play(self, valueFunc, rounds=100):
        for _ in range(rounds):
            self.reset()
            action = self.chooseAction()
            while not self.end:
                nxtState = self.takeAction(action)
                self.reward = self.giveReward(nxtState)
                valueFunc.updateZ(self.state)
                valueFunc.learn(self.state, nxtState, self.reward)
                self.state = nxtState
                action = self.chooseAction()
                if self.debug:
                    print("end at {} reward {}".format(self.state, self.reward))
actual_state_values = np.arange(-20, 22, 2) / 20.0


actual_state_values[0] = actual_state_values[-1] = 0

alphas = np.linspace(0, 0.8, 6)
lambdas = np.linspace(0, 1, 5)
rounds = 50

plt.figure(figsize=[10, 6])
for lamb in lambdas:
    alpha_errors = []
    for alpha in alphas:
        valueFunc = ValueFunctionTD(alpha=alpha, lmbda=lamb)
        rw = RWTD(debug=False)
        rw.play(valueFunc, rounds=rounds)
        rmse = np.sqrt(np.mean(np.power(valueFunc.weights - actual_state_values, 2)))
        print("lambda {} alpha {} rmse {}".format(lamb, alpha, rmse))
        alpha_errors.append(rmse)
    plt.plot(alphas, alpha_errors, label="lambda={}".format(lamb))

plt.xlabel("alpha", size=14)
plt.ylabel("RMS error", size=14)
plt.legend()
plt.show()
