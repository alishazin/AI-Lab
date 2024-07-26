
import time
import random

class VaccumCleanerAgent:

    actions_map = {
        ('A', 'clean'): 'moveRight',
        ('B', 'clean'): 'moveLeft',
        ('A', 'dirty'): 'suck',
        ('B', 'dirty'): 'suck',
    }

    def __init__(self, location='A'):
        self.location = location
        self.status = self.get_random_state()

    def percept(self):
        """ This function simply returns the current status. (Since it is a simple reflex agent) """
        
        return self.status
    
    def get_random_state(self):
        """ Returns either 'clean' or 'dirty' with a probability of 1/2 each """

        return 'clean' if random.randint(0,1) else 'dirty' 
    
    def act(self, action, logging=False):
        """ Function called to take an action. Action can be one of [moveRight, moveLeft, suck] """

        # If logging is True, current location, status and action taken will be printed
        if logging:
            print(f"Location: Room {self.location}")
            print(f"Percept: {self.status}")
            print(f"Action Taken: {action}\n")
        
        if action == 'moveRight':
            self.location = 'B'
        elif action == 'moveLeft':
            self.location = 'A'
        elif action == 'suck':
            self.status = 'clean'

    def print_instructions(self):
        """ It simply print out the instructions to follow when simulating the vaccum cleaner """

        print("Vaccum cleaner is switched ON.\nTo switch it OFF, press Ctrl + C\n")

    def simulate(self, randomize_env=True):
        """ Used to simulate the working of the vaccum cleaner. Set randomize_env to True if you want the room to become dirty implicitly during the
        simulation """

        self.print_instructions()

        while True:

            try:

                current_state = (self.location, self.status)
                action = self.actions_map.get(current_state, "INVALID")

                if action == "INVALID":
                    print("Vaccum cleaner has entered to an INVALID state, and it doesn't know what to do")
                    return None
                
                self.act(action, True)

                if action in ['moveRight', 'moveLeft'] and randomize_env:
                    self.status = self.get_random_state()

                time.sleep(2)
            
            except KeyboardInterrupt:

                print("Vaccum cleaner is switched OFF.\nThank You")
                return None

vc = VaccumCleanerAgent()

vc.simulate()