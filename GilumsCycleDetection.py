from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices #Number of vertices
        self.graph = defaultdict(list) #storing the graph

    def addEdge(self, u, v):
        self.graph[u].append(v)

    #helper function to find the subset of an element i
    def find_parent(self, parent, i):
        if parent[i] == -1:
            return i
        if parent[i] != -1:
            return self.find_parent(parent, parent[i])
    
    #helper function to do union on two subsets
    def union(self, parent, x, y):
        parent[x] = y

    #checks if graph contains cycles or not
    def isCyclic(self):

        #to create the list of -1 in all
        #for each Vertex
        parent = [-1] * (self.V)

        #goes through all edges of the graph
        #finds the subset, if two subsets
        #share the same name, then there's 
        #a cycle
        for i in self.graph: #each vertex in graph
            for j in self.graph[i]: #each point that vertex connects
                x = self.find_parent(parent, i)
                y = self.find_parent(parent, j)
                if x == y:
                    return True
                self.union(parent, x, y)


g = Graph(3)
g.addEdge(0, 1)
g.addEdge(1, 2)
g.addEdge(2, 0)
 
if g.isCyclic():
    print("Graph contains cycle")
else :
    print("Graph does not contain cycle ")
