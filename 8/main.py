# Solve TSP using Dynamic Programing

# Formula: g(i, S) = min { 
#       ( 
#           C[i][k] + g(k, S - {k}) 
#       ) for all k ε S 
# }

# where C = Adjacency Matrix
#       S = Set of all unvisited neighbours
#       i = initially, starting node

class TravellingSalesmanProblem:

    def __init__(self, matrix):
        self.M = matrix
        self.all = set(range(0, len(matrix)))
        # 1st row is considered as the starting node

    def solve(self):
        return self.g(0, self.all - set([0]))

    def g(self, i, S):

        if len(S) == 0:
            tempValue = self.M[i][0]
            path = [i+1, 1]
            return [path, tempValue]
        
        minIndex = None
        minValue = None
        count = 0
        paths = []
        
        for k in S:
            
            path, tempValue = self.g(k, S - set([k]))
            tempValue += self.M[i][k] 
            
            paths.append(path)
            
            if minValue == None or minValue > tempValue:
                minValue = tempValue
                minIndex = count
            count += 1
        
        path = paths[minIndex]
        path.insert(0,i+1)
        
        return [path, minValue]

# tsp = TravellingSalesmanProblem([
#     [0 , 10, 15, 20], 
#     [10, 0 , 25, 25], 
#     [15, 25, 0 , 30],
#     [20, 25, 30, 0 ]
# ])

tsp = TravellingSalesmanProblem([
    [0 , 15, 5, 20], 
    [10, 0 , 25, 25], 
    [15, 25, 0 , 30],
    [20, 25, 30, 0 ]
])

# tsp = TravellingSalesmanProblem([
#     [0 , 10, 15, 20],
#     [5 , 0 , 9 , 10],
#     [6 , 13, 0 , 12],
#     [8 , 8 , 9 , 0 ],
# ])

path, cost = tsp.solve()

print("Cost: ", cost)

print("Path: ", *(f"{i} -> " for i in path[0:-1]), end="")
print(path[-1]) 