import time

actions = {
    0: "Fill A", 
    1: "Fill B", 
    2: "Empty B", 
    3: "Empty B", 
    4: "Pour from B to A", 
    5: "Pour from A to B", 
}

def water_jug_problem(capacity1, capacity2, target, initial1=0, initial2=0):
    
    # initial state (x, y) where x and y are the amounts of water in the two jugs
    state = (initial1, initial2)
    parent = {}
    frontier = [state]
    
    while frontier:
        
        state = frontier.pop(0)
        x, y = state

        # generate all possible successor states
        states = [
            ((capacity1, y), 0), 
            ((x, capacity2), 1), 
            ((0, y), 2), 
            ((x, 0), 3), 
            ((min(x + y, capacity1), max(0, x + y - capacity1)), 4), 
            ((max(0, y + x - capacity2), min(y + x, capacity2)), 5)
        ]
        for new_state in states:
            
            if new_state[0] not in parent:
                
                # To store the path to reach the state
                parent[new_state[0]] = (state, new_state[1])

                # Queue from which th next state is popped
                frontier.append(new_state[0])
                
                if new_state[0] == (target, 0) or new_state[0] == (0, target): 
                    # Backtracking till the starting state, to find the path

                    cur_state = new_state[0]

                    path = [cur_state]
                    while (cur_state in parent) and (cur_state != (initial1, initial2)):
                        cur_state = parent[cur_state]
                        path.append(cur_state)
                        cur_state = cur_state[0]
                    
                    path.reverse()
                    return path
                
    return None

# example
capacity1 = int(input("Enter Capacity Of Jug 1: "))
capacity2 = int(input("Enter Capacity Of Jug 2: "))
target = int(input("Enter Target: "))
initial1 = int(input("Enter Initial Quantity Of Jug 1: "))
initial2 = int(input("Enter Initial Quantity Of Jug 2: "))

path = water_jug_problem(capacity1, capacity2, target, initial1, initial2)
if path is None:
    print("No solution found.")
else:
    print("\nSequence")
    for state in path:
        if (type(state[0]) != int):
            print(f"State: {(state[0][0], state[0][1])}, Next Action: {actions[state[1]]}")
        else:
            print(f"State: {(state[0], state[1])}, Goal State Reached!")
