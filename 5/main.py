
class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class Tree:

    def __init__(self, tree):
        self.tree = tree

    # DFS Traversal(s)

    def pre_order(self):

        def _helper(root):

            if root is None:
                return
            
            print(root.value, end=" ")
            _helper(root.left)
            _helper(root.right)

        _helper(self.tree)
        print()

    def in_order(self):

        def _helper(root):

            if root is None:
                return
            
            _helper(root.left)
            print(root.value, end=" ")
            _helper(root.right)

        _helper(self.tree)
        print()

    def post_order(self):

        def _helper(root):

            if root is None:
                return
            
            _helper(root.left)
            _helper(root.right)
            print(root.value, end=" ")

        _helper(self.tree)
        print()

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

print("Pre Order: ", end="")
tree.pre_order()

print("In Order: ", end="")
tree.in_order()

print("Post Order: ", end="")
tree.post_order()

print("Level Order: ", end="")
tree.level_order()
