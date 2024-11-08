from queue import Queue
import math

def tsp_bfs(graph, start=0):
    num_cities = len(graph)
    # The queue will store tuples of (current_city, path_visited, cost, path)
    queue = Queue()
    # Start BFS with the initial city
    queue.put((start, [start], 0, [start]))
    min_cost = math.inf
    best_path = []

    while not queue.empty():
        current_city, path_visited, cost, path = queue.get()

        # If all cities are visited, check if it forms a better solution
        if len(path_visited) == num_cities:
            # Return to the start to complete the cycle
            total_cost = cost + graph[current_city][start]
            if total_cost < min_cost:
                min_cost = total_cost
                best_path = path + [start]
            continue

        # Visit the neighboring cities
        for next_city in range(num_cities):
            if next_city not in path_visited:
                next_cost = cost + graph[current_city][next_city]
                queue.put((next_city, path_visited + [next_city], next_cost, path + [next_city]))

    return min_cost, best_path

# Example graph (adjacency matrix)
graph = [
    [0 , 15, 5, 20], 
    [10, 0 , 25, 25], 
    [15, 25, 0 , 30],
    [20, 25, 30, 0 ]
]

# Starting from city 0
min_cost, best_path = tsp_bfs(graph)
print(f"Minimum cost: {min_cost}")
print(f"Best path: {best_path}")
