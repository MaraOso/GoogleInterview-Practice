class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_level(self):
        level = 0
        p = self.parent
        while p:
            level += 1
            p = p.parent

        return level

    def print_tree(self):
        spaces = ' ' * self.get_level() * 4
        prefix = spaces + "|___" if self.parent else ""
        print(prefix + self.data)
        if self.children:
            for child in self.children:
                child.print_tree()

def build_product_tree():
    root = TreeNode("Electronics")
    laptop = TreeNode("Laptop")
    catWatch = TreeNode("Cat Watch")
    nekoWatch = TreeNode("Neko Wath")

    root.add_child(laptop)
    root.add_child(catWatch)
    catWatch.add_child(nekoWatch)

    return root

class BinarySearchTreeNode:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

    def add_child(self, data):
        if data == self.data:
            return
        if data < self.data:
            if self.left:
                self.left.add_child(data)
            else:
                self.left = BinarySearchTreeNode(data)
        else:
            if self.right:
                self.right.add_child(data)
            else:
                self.right = BinarySearchTreeNode(data)

    def in_order_traversal(self):
        elements = []
        if self.left:
            elements += self.left.in_order_traversal()

        elements.append(self.data)

        if self.right:
            elements += self.right.in_order_traversal()

        return elements

    def search(self, val):
        if self.data == val:
            return True
        if val < self.data:
            if self.left:
                return self.left.search(val)
            else:
                return False
        if val > self.data:
            if self.right:
                return self.right.search(val)
            else:
                return False

    def find_max(self):
        if self.right is None:
            return self.data
        return self.right.find_max()

    def find_min(self):
        if self.left is None:
            return self.data
        return self.left.find_min()

    def delete(self, val):
        if val < self.data:
            if self.left:
                self.left = self.left.delete(val)
        elif val > self.data:
            if self.right:
                self.right = self.right.delete(val)

        else:
            if self.left is None and self.right is None:
                return None
            if self.left is None:
                return self.right
            if self.right is None:
                return self.left

            min_val = self.right.find_min()
            self.data = min_val
            self.right = self.right.delete(min_val)

        return self


def build_tree(elements):
    root = BinarySearchTreeNode(elements[0])
    
    for i in range(1, len(elements)):
        root.add_child(elements[i])

    return root


if __name__ == '__main__':
    #root = build_product_tree()
    #root.print_tree()
    numbers = [17,4,1,20,9,23,18,34]
    numbers_tree = build_tree(numbers)
    numbers_tree.delete(23)
    print(numbers_tree.search(23))


import sys

graph = {'a': {'b': 10, 'c': 13}, 'b': {'c': 1, 'd': 2}, 'c': {'b': 4, 'd':8, 'e': 12}, 'd': {'e': 7}, 'e': {'d': 9}}

def djKstra(graph, start, goal):
    shorted_distance = {}
    predessesor = {}
    unseenNodes = graph
    infinity = sys.maxsize
    path = []

    for node in unseenNodes:
        shorted_distance[node] = infinity
    shorted_distance[start] = 0

    while unseenNodes:
        minNode = None
        for node in unseenNodes:
            if minNode == None:
                minNode = node
            elif shorted_distance[node] < shorted_distance[minNode]:
                minNode = node

        for childNode, weight in graph[minNode].items():
            if weight + shorted_distance[minNode] < shorted_distance[childNode]:
                shorted_distance[childNode] = weight + shorted_distance[minNode]
                predessesor[childNode] = minNode
        unseenNodes.pop(minNode)

    currentNode = goal
    while currentNode != start:
        try:
            path.insert(0, currentNode)
            currentNode = predessesor[currentNode]
        except KeyError:
            print("Path not found")
            break
    path.insert(0,start)
    if shorted_distance[goal] != infinity:
        print('Shortest distance is: ' + str(shorted_distance[goal]))
        print('Shortest path is: ' + str(path))


    print(shorted_distance)
        

djKstra(graph, 'e', 'a')
