
# f(x) = g(x) + h(x)
# g(x) = actual cost
# h(x) = heuristic value

# Function to get the index of the node with least f(x) from open_list
def getIndexForLowestFX(open_list):

    minValue = float('inf')
    minIndex = None

    for i in range(len(open_list)):
        if open_list[i][1] < minValue:
            minValue = open_list[i][1]
            minIndex = i

    return minIndex

# Best First Search
def bestFirstSearch(graph, heuristics, source, target):

    # node, f(x), sum of g(x), path
    open_list = [[source, 0, 0, []]]
    closed_list = []

    while open_list:

        temp_node, temp_fx, act_cost_sum, temp_path = open_list.pop(getIndexForLowestFX(open_list))

        if temp_node == target:
            return {'cost': act_cost_sum, 'path': [*temp_path, temp_node]}
        
        if temp_node in closed_list: continue

        for node, act_cost in graph[temp_node]:
            
            open_list.append([node, heuristics[node], act_cost_sum + act_cost, [*temp_path, temp_node]])

        closed_list.append(temp_node)

    return None


graph = {
    'P': [['R', 4], ['C', 4], ['A', 4]],
    'R': [['E', 5]],
    'C': [['M', 6], ['U', 3], ['R', 2]],
    'A': [['M', 3]],
    'E': [['U', 5], ['S', 1]],
    'M': [['U', 5], ['L', 2]],
    'U': [['N', 5], ['S', 4]],
    'L': [['N', 5]],
    'N': [['S', 6]],
    'S': [],
}

heuristics = {
    'P': 10,
    'R': 8,
    'C': 6,
    'A': 11,
    'E': 3,
    'M': 9,
    'U': 4,
    'L': 9,
    'N': 6,
    'S': 0,
}

result = bestFirstSearch(graph, heuristics, 'P', 'S')

if result:
    print("Actual Cost:", result['cost'])
    
    print("Path: ", end="")
    for i in result['path'][0:-1]:
        print(i, " -> ", end="")
    print(result['path'][-1])
else:
    print("Couldn't find a path.")