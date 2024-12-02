import math

actions = {
    0: "Fill A",
    1: "Fill B",
    2: "Empty A",
    3: "Empty B",
    4: "Pour B to A",
    5: "Pour A to B",
}

def water_jug(c1, c2, target, init1, init2):

    if (target % math.gcd(c1, c2) != 0 or (c1 > target and c2 > target)):
        print("Can't find the solution.")
        return -1
    
    frontier = [(init1, init2)]
    parent = {}

    while len(frontier):

        state = frontier.pop(0)
        x, y = state

        new_states = [
            [(c1, y), 0],
            [(x, c2), 1],
            [(0, y), 2],
            [(x, 0), 3],
            [(max(c1, x+y), y - (c1-x)), 4],
            [(x - (c2-y), max(c2, x+y)), 5],
        ]

        for new_state in new_states:

            if new_state[0] not in parent and new_state[0] not in frontier:

                parent[new_state[0]] = [state, new_state[1]]
                frontier.append(new_state[0])

                if new_state[0] == (target, 0) or new_state[0] == (0, target):

                    cur_state = new_state
                    path = [ [cur_state[0], None] ]

                    while cur_state[0] != (init1, init2):
                        cur_state = parent[cur_state[0]]
                        path.insert(0, cur_state)

                    return path
                

path = water_jug(2, 5, 3, 1, 0)

for state, next_action in path:

    print("State: ", state)

    if next_action:
        print("Action Taken: ", actions[next_action])
    else:
        print("Final state is reached.")