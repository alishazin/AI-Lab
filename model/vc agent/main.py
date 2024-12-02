import random
import time

class VaccumCleanerAgent:

    action_map = {
        ('A', 'clean'): 'moveRight',
        ('B', 'clean'): 'moveLeft',
        ('A', 'dirty'): 'suck',
        ('B', 'dirty'): 'suck',
    }

    def __init__(self, location='A'):
        self.state = (location, self.get_random_state())

    def get_random_state(self):
        return "clean" if random.randint(0, 1) == 0 else "dirty"

    def act(self, action):

        print(f"Current State: {self.state}")
        print(f"Action Taken: {action}")

        if action == 'moveRight': 
            self.state = ('B', self.state[1])
        elif action == 'moveLeft': 
            self.state = ('A', self.state[1])
        elif action == 'suck': 
            self.state = (self.state[0], "clean")

    def simulate(self):

        while True:

            action = self.action_map[self.state]

            self.act(action)

            if action in ['moveRight', 'moveLeft']:
                self.state = (self.state[0], self.get_random_state())
            
            time.sleep(2)


vc = VaccumCleanerAgent()

vc.simulate()