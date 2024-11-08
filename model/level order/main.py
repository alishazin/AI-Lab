
class Node:

    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def level_order(tree):

    def height(node):

        if node is None: return 0

        lh = height(node.left)
        rh = height(node.right)

        if lh > rh:
            return lh + 1
        else:
            return rh + 1
        
    def printCurrentLevel(node, level):

        if node is None: return

        if level == 1: print(node.value, end=', ')
        else:
            printCurrentLevel(node.left, level-1)
            printCurrentLevel(node.right, level-1)

    for i in range(1, height(tree) + 1):
        printCurrentLevel(tree, i)


tree = Node('A', 
    Node('B', 
        Node('C'),    
        Node('D'),    
    ),
    Node('E', 
        Node('F'),    
        Node('G'),    
    )
)

level_order(tree)