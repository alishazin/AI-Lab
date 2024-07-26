

def dfs(graph, root):

    visited = set()

    def inner_func(graph, root):
    
        visited.add(root)
        print(root)

        print(graph[root] - visited)
        print("-----")
        for next in graph[root] - visited:
            if next not in visited:
                inner_func(graph, next)

    inner_func(graph, root)

graph = {
    'A' : set(['B', 'D', 'F']),
    'B' : set(['D', 'C']),
    'C' : set([]),
    'D' : set(['E', 'F']),
    'E' : set([]),
    'F' : set([]),
}

dfs(graph, 'A')

"""
A
B D F 
C E
"""