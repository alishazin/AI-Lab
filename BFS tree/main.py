
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class Tree:

    def __init__(self, tree):
        self.tree = tree

    # BFS Traversal(s)

    def level_order(self):

        def height(node):

            if node is None:
                return 0
            
            lheight = height(node.left)
            rheight = height(node.right)

            if lheight > rheight:
                return lheight + 1
            else:
                return rheight + 1
            
        def printCurrentLevel(node, level):

            if node is None:
                return
            
            if level == 1:
                print(node.value, end=" ")
            elif level > 1:
                printCurrentLevel(node.left, level-1)
                printCurrentLevel(node.right, level-1)

        for i in range(1, height(self.tree) + 1):
            printCurrentLevel(self.tree, i)

        print()
        

tree = Tree(
    Node('A', 
        Node('B', 
            Node('C'),    
            Node('D'),    
        ),
        Node('E', 
            Node('F'),    
            Node('G'),    
        )
    )
)


print("Level Order: ", end="")
tree.level_order()
