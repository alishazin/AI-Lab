
class NQueenProblem:

    def __init__(self, n):
        
        self.N = n
        self.board = [[0 for j in range(self.N)] for i in range(self.N)]


    def printBoard(self):
        
        for i in range(self.N):
            for j in range(self.N):
                print(".  " if self.board[i][j] == 0 else "Q  ", end="")
            print("\n")


    def isSafe(self, row, col):

        # Check Same Column and Row
        for i in range(self.N):
            if self.board[i][col] == 1:
                return False
            if self.board[row][i] == 1:
                return False

        # Check Diagonal (Top Left to Bottom Right)
        for i, j in zip(
                range(row - min(row, col), row + (self.N - max(row, col))), 
                range(col - min(row, col), col + (self.N - max(row, col)))
            ):
            if self.board[i][j] == 1: return False

        # Check Diagonal (Bottom Left to Top Right)
        for i, j in zip(
                range(row + min(self.N - 1 - row, col), row - min(self.N - 1 - col, row) - 1, -1),
                range(col - min(self.N - 1 - row, col), col + min(self.N - 1 - col, row) + 1)
            ):
            if self.board[i][j] == 1: return False

        return True


    def solve(self, row=0):

        if row >= self.N:
            return True
        
        for i in range(self.N):

            if self.isSafe(row, i):

                self.board[row][i] = 1

                if self.solve(row+1) == True:
                    return True

                self.board[row][i] = 0
        
        return False


obj = NQueenProblem(int(input("Enter N value: ")))

if obj.solve():
    obj.printBoard()
else:
    print("Solution does not exist")