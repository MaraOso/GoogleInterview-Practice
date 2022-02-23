from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []

    def addEdge(self, u, v, w): #w is the dist between
        self.graph.append([u,v,w])

    #helper function to find set of element i
    #uses path compression technique
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    #forms a union of two sets of x and y
    #uses union by rank
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        #attaches smaller ranked tree under root
        #high rank tree (union by rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        #if they are equal then one is made a root
        #and increase its rank by 1
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def kruskalMST(self):
        result = []
        i = 0 #index list to store sorted edges
        e = 0 #index list to store result[]

        #step1: sort all the edges in
        #non-decreasing order of their
        #weight. if you're not allowed to change
        #the graph create a copy.
        self.graph = sorted(self.graph, key=lambda item: item[2])
        #sorted by the [2] in the Vertices which is weight

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:

            #step2: Pick the smallest edge and increment
            #the index for next iteration
            u,v,w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            #if the edge does cause a cycle
            #include it in result and increment
            #to the next edge
            if x != y:
                e = e +1
                result.append([u,v,w])
                self.union(parent, rank, x, y)
            #else discared the edge

            minimumCost = 0
            print("Edges in the constructed MST")
            for u, v, weight in result:
                minimumCost += weight
                print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree" , minimumCost)

g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)
 
# Function call
g.kruskalMST()