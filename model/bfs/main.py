
def bs(graph, source):

    queue = [source]
    visited = []

    while len(queue):

        root = queue.pop(0)
        print(root, end=", ")
        visited.append(root)

        for child in graph[root]:
            if child not in visited and child not in queue:
                queue.append(child)

def bfs(graph, startNode):
    
    visited = []
    queue = [startNode]

    while True:
        
        if len(queue) == 0:
            break

        startNode = queue.pop(0)
        print(startNode, end=", ")
        visited.append(startNode)

        for i in graph[startNode]:
            if i not in visited and i not in queue:
                queue.append(i)

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['E', 'F'],
    'C': [],
    'D': ['G', 'H'],
    'E': [],
    'F': [],
    'G': [],
    'H': [],
}

bfs(graph, 'A')

"""
        A
  B     C     D
E   F       G   H
"""