

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def level_order(tree):

    def height(node):

        if node is None:
            return 0
        
        lh = height(node.left)
        rh = height(node.right)

        return max(lh, rh) + 1
    
    def printLevel(node, level):

        if node is None:
            return
        
        if level == 0:
            print(node.value)
        else:
            printLevel(node.left, level-1)
            printLevel(node.right, level-1)


    for i in range(height(tree)):
        printLevel(tree, i)


tree = Node('A', 
    Node('B', 
        Node('C'),    
        Node('D'),    
    ),
    Node('E', 
        Node('F', Node('H')),    
        Node('G', Node('I')),    
    )
)

level_order(tree)