import math

actions = {
    0: 'Fill A',
    1: 'Fill B',
    2: 'Empty A',
    3: 'Empty B',
    4: 'Pour B to A',
    5: 'Pour A to B',
}

def water_jug_problem(c1, c2, target, init1, init2):

    if (target % math.gcd(c1, c2) != 0) or ((target > c1 and target > c2)):
        print("No solution can be found.")
        return

    frontier = [(init1, init2)]
    parent = {}

    while frontier:

        state = frontier.pop(0)
        x, y = state

        new_states = [
            [(c1, y), 0],
            [(x, c2), 1],
            [(0, y), 2],
            [(x, 0), 3],
            [(min(c1, x + y), y - (c1 - x)), 4],
            [(x - (c2 - y), min(c2, y + x)), 5],
        ]

        for new_state in new_states:

            if new_state[0] not in parent and new_state[0] not in frontier:

                parent[new_state[0]] = [state, new_state[1]]
                frontier.append(new_state[0])

                if new_state[0] == (target, 0) or new_state[0] == (0, target):

                    cur_state = new_state
                    path = [(cur_state[0], None)]

                    while cur_state[0] != (init1, init2):
                        cur_state = parent[cur_state[0]]
                        path.insert(0, cur_state)

                    return path
                
path = water_jug_problem(2, 5, 3, 1, 0)

for state, action in path:
    print("State: ", state)
    if action == None:
        print("Final State is Reached.")
    else:
        print("Next Action: ", actions[action])
    print()
