import itertools

def calculate_total_distance(path, distances):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distances[(path[i], path[i+1])]
    total_distance += distances[(path[-1], path[0])]
    return total_distance

def bfs_tsp(landmarks, distances):
    queue = [(landmarks[0], [landmarks[0]])]
    min_path = None
    min_distance = float("inf")
    while queue:
        current_landmark, path = queue.pop(0)
        if len(path) == len(landmarks):
            path_distance = calculate_total_distance(path, distances)
            if path_distance < min_distance:
                min_distance = path_distance
                min_path = path
        else:
            for next_landmark in landmarks:
                if next_landmark not in path:
                    new_path = path + [next_landmark]
                    queue.append((next_landmark, new_path))
    return min_path, min_distance

if __name__ == "__main__":
    num_landmarks = int(input("Enter the number of landmarks: "))
    landmarks = []
    for i in range(num_landmarks):
        landmark = input(f"Enter the name of landmark {i+1}: ")
        landmarks.append(landmark)
    distances = {}
    print("Enter the distances between the landmarks (in km):")
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = float(input(f"Distance from {landmarks[i]} to {landmarks[j]}: "))
            distances[(landmarks[i], landmarks[j])] = dist
            distances[(landmarks[j], landmarks[i])] = dist
    print("\nSolving TSP using BFS:")
    bfs_solution, bfs_distance = bfs_tsp(landmarks, distances)
    print(f"Path: {bfs_solution}, Distance: {bfs_distance}")