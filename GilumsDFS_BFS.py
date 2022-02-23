###Gilums - Graphs

from collections import defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    #add edge to graph from 'u' to 'v'
    def addEdge(self, u, v):
        self.graph[u].append(v)

    def BFS(self, s):
        visited = [False] * (max(self.graph) + 1)

        #creates a queue for BFS
        queue = []

        #mark the sources node 's' as visited
        #and enqueue it
        queue.append(s)
        visited[s] = True

        #removes vertex from search and prints it
        while queue:
            s = queue.pop(0)
            print(s, end= " ")

            #gets adjacent verts of the removed vertex
            #if they weren't visited then they are
            #marked as visited and enqueued
            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True
    
    #helper function for DFS
    def DFSUtil(self, v, visited):

        #marks the current node as visited
        visited.add(v)
        print(v, end= " ")

        #recur for all vertices
        #adjacent to this vertex 'v'
        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self.DFSUtil(neighbor, visited)

    #uses DFSUtil
    def DFS(self, v):
        
        #create a set to store vertices
        visited = set()

        #calls the recursive helper function
        for vertex in list(self.graph):
            if vertex not in visited:
                self.DFSUtil(v, visited)

        


g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
 
print ("Following is Breadth First Traversal"
                  " (starting from vertex 2)")
g.DFS(2)