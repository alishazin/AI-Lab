
def dfs(graph, source):

    visited = set()

    def inner(node):

        print(node)
        visited.add(node)

        for i in graph[node]:

            if i not in visited:
                inner(i)

    inner(source)

graph = {
    '1' : ['2', '3'],
    '2' : ['4', '5'],
    '3' : ['6', '7'],
    '4' : ['6'],
    '5' : [],
    '6' : [],
    '7' : [],
}

dfs(graph, '1')