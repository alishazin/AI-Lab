
# evaluation function
def f(source, dest):
    pass

# Function to get the index of the node with least f(x) from open_list
def getIndexForLowestFX(open_list):

    minValue = float('inf')
    minIndex = None

    for i in range(open_list):
        if open_list[i][1] < minValue:
            minValue = open_list[i][1]
            minIndex = i

    return minIndex


def aStarSearch(graph, heuristics, source, target):

    # node, f(x), sum of actual_cost
    open_list = [[source, 0, 0]]
    closed_list = [source]

    while open_list:

        temp_node, temp_fx, act_cost_sum = open_list.pop(getIndexForLowestFX(open_list))

        if temp_node == target:
            return [temp_node, temp_fx, act_cost_sum]
        
        # exclude if from closed_list

        for node, act_cost in graph[temp_node]:

            open_list.append([node, act_cost_sum + act_cost + heuristics[node], act_cost_sum + act_cost])

        closed_list.append(temp_node)

    return []


graph = {
    'S': [['A', 4], ['B', 10], ['C', 11]],
    'A': [['B', 8], ['D', 5]],
    'B': [['D', 15]],
    'C': [['D', 8], ['F', 2], ['E', 20]],
    'D': [['H', 16], ['I', 20], ['F', 1]],
    'E': [['G', 19]],
    'F': [['G', 13]],
    'G': [],
    'H': [['I', 1], ['J', 2]],
    'I': [['J', 5], ['K', 13], ['G', 5]],
    'J': [['K', 7]],
    'K': [['G', 16]]
}

heuristics = {
    'S': 7,
    'A': 8,
    'B': 6,
    'C': 5,
    'D': 5,
    'E': 3,
    'F': 3,
    'G': 0,
    'H': 7,
    'I': 4,
    'J': 5,
    'K': 3
}