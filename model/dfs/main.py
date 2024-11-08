
def dfs(graph, source):

    visited = set()

    def inner(root):

        visited.add(root)
        print(root, end=", ")
        
        for child in graph[root]:

            if child not in visited:
                inner(child)

    inner(source)



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

dfs(graph, 'A')

"""
        A
  B     C     D
E   F       G   H
"""