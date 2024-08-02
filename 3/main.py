
def bfs(graph, startNode):
    
    visited = []
    queue = [startNode]

    while True:
        
        if len(queue) == 0:
            break

        startNode = queue.pop(0)
        print(startNode)
        visited.append(startNode)

        for i in graph[startNode]:
            if i not in visited and i not in queue:
                queue.append(i)

graph = {
    0: [1,2],
    1: [3,2],
    2: [1,4],
    3: [1,4],
    4: [2,3],
}

bfs(graph, 0)