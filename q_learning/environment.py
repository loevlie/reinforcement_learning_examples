import numpy as np
import matplotlib.pyplot as plt
import random
import numpy as np

class env:
    def __init__(self, discount_factor = 0.5):
        self.states = ["s1", "s2", "s3","s4","s5","s6"]
        self.actions = {
            "s1":["s2", "s4"],
            "s2": ["s3", "s5"],
            "s3": ["s2", "s6"],
            "s4": ["s1", "s5"],
            "s5":["s4","s6"],
            "s6":None
        }
        self.reward = {"s1":0,"s2":0,"s3":0,"s4":0,"s5":0,"s6":100}
        self.q_table = {
            "s1":{"s2":0,"s4":0},
            "s2": {"s1":0, "s3":0, "s5":0},
            "s3": {"s2":0, "s6":0},
            "s4": {"s1":0, "s5":0},
            "s5": {"s3":0, "s4":0, "s6":0},
            "s6": {"none":0}
        }
        self.current_state = "s1"
        self.disc_f = discount_factor

    def list_actions(self):
        return self.actions[self.current_state]

    def list_qtable_values(self):
        return self.q_table[self.current_state]

    def update_table(self, next_state):
        self.q_table[self.current_state][next_state] = (
            self.reward[next_state]
            + self.disc_f
            * max(self.q_table[next_state].values())
        )

    def take_action(self, s2):
        assert s2 in self.actions[self.current_state], (
            f"This action is not in the current list of possible actions "
            f"{self.actions[self.current_state]}"
        )
        self.update_table(s2)
        if self.actions[s2]:
            self.current_state = s2
        else:
            self.current_state = "s1"
        return self.reward[s2]

if __name__ == "__main__":
    environment = env()
    print(f"Current actions: {environment.list_actions()}")
    epsilon = 0.1
    for i in range(100):
        if random.random() > (1-epsilon):
            # Exploiting
            next_action = environment.list_actions()[np.argmax(environment.list_qtable_values())]
        else:
            # Exploring
            next_action = random.choice(environment.list_actions())

        print(f"Action - {environment.current_state} to {next_action}")
        reward = environment.take_action(next_action)
        print(f"Iteration {i} / 100 - Reward returned - {reward}")
        print(environment.q_table)
        print()
