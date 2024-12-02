
def bfs(graph, source):

    visited = []
    queue = [source]

    while len(queue) != 0:

        node = queue.pop(0)
        print(node)
        visited.append(node)

        for i in graph[node]:
            if i not in visited and i not in queue:
                queue.append(i)


graph = {
    '1': ['2', '3'],
    '2': ['5', '4'],
    '3': ['4'],
    '4': ['6'],
    '5': ['6'],
    '6': [],
}
# graph = {
#     '1' : ['2', '3'],
#     '2' : ['4', '5'],
#     '3' : ['6', '7'],
#     '4' : [],
#     '5' : [],
#     '6' : [],
#     '7' : [],
# }


bfs(graph, '1')