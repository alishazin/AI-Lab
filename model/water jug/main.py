
import math

def solve_water_jug(c1, c2, target):

    if (target % math.gcd(c1, c2) != 0) or ((target > c1 and target > c2)):
        print("Not possible according to Bézout's identity.")
        return -1

    state = (0, 0)
    frontier = [state]
    parent = {}

    while frontier:

        state = frontier.pop(0)
        x, y = state

        states = [
            (c1, y),
            (x, c2),
            (0, y),
            (x, 0),
            (min(c1, x+y), max(0, x + y - c1)),
            (max(0, x+y-c2), min(c2, x+y)),
        ]

        for new_state in states:

            if new_state not in parent:

                parent[new_state] = state
                frontier.append(new_state)

                if new_state == (target, 0) or new_state == (0, target):

                    cur_state = new_state

                    path = [cur_state]

                    while (cur_state in parent) and cur_state != (0, 0):
                        cur_state = parent[cur_state]
                        path.insert(0, cur_state)

                    return path


print(solve_water_jug(5, 3, 4))