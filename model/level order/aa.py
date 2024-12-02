
class Node:

    def __init__(self, value, left=None, right=None):

        self.value = value
        self.left = left
        self.right = right


class Tree:

    def __init__(self, tree):
        self.root = tree

    def pre_order(self):

        def _helper(root):

            if root is None:
                return
            
            print(root.value, end=" ")
            _helper(root.left)
            _helper(root.right)

        _helper(self.root)
    
    def in_order(self):

        def _helper(root):

            if root is None:
                return
            
            _helper(root.left)
            print(root.value, end=" ")
            _helper(root.right)

        _helper(self.root)
    
    def post_order(self):

        def _helper(root):

            if root is None:
                return
            
            _helper(root.left)
            _helper(root.right)
            print(root.value, end=" ")

        _helper(self.root)

    def level_order(self):

        def height(root):

            if root is None:
                return 0
            
            lh = height(root.left)
            rh = height(root.right)

            return max(lh, rh) + 1
        
        def printLevel(root, level):

            if root is None:
                return
            
            if level == 0:
                print(root.value, end=" ")
            else:
                printLevel(root.left, level-1)
                printLevel(root.right, level-1)

        for i in range(height(self.root)):
            printLevel(self.root, level=i)


tree = Tree(Node('A', 
    Node('B', 
        Node('D'),
        Node('E'),
    ),
    Node('C',
        Node('F'),
        Node('G'),
    ),
))
"""
     A
  B      C
D   E  F   G
"""

tree.pre_order()
print()

tree.in_order()
print()

tree.post_order()
print()

tree.level_order()
print()