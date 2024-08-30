
class TravellingSalesmanProblem:

    def __init__(self, matrix):
        self.M = matrix
        self.all = set(range(0, len(matrix)))
        # 1st row is implicitly considered as the starting point

    def solve(self):
        path = []
        return [path, self.g(path, 0, self.all - set([0]))]

    def g(self, path, i, S):

        if len(S) == 0:
            tempValue = self.M[i][0]
            path.append([i+1, tempValue])
            return tempValue
        
        minK = None
        minValue = None
        for k in S:
            tempValue = self.M[i][k] + self.g(path, k, S - set([k]))
            if minValue == None or minValue > tempValue:
                minValue = tempValue
                minK = k
        
        path.append([minK+1, minValue])
        
        return minValue

tsp = TravellingSalesmanProblem([
    [0 , 10, 15, 20],
    [5 , 0 , 9 , 10],
    [6 , 13, 0 , 12],
    [8 , 8 , 9 , 0 ],
])

print(tsp.solve())
 