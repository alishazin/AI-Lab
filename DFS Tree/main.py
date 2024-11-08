
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
