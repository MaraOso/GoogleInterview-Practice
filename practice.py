graph = [(5,2), (5,0), (4,0), (4,1), (2,3), (3,1), (1,2)]
weightedGraph = [(0,1,8), (0,4,3), (5,3,1), (2, 1,6), (1,4,3), (3,4,5), (4,5,2)]

from collections import defaultdict
import heapq as heap 

class Graph:
    def __init__(self, verts):
        self.adjList = defaultdict(list)
        self.time = 0
        self.adjMatrix = []

    def countVerts(self, verts, bellman= False):
        if not bellman:
            count = set()
            for u,v in verts:
                count.add(u)
                count.add(v)
            return len(count)
        else:
            count = set()
            for u,v,w in verts:
                count.add(u)
            return len(count)

    def createAdjList(self, verts, directed= False, weighted= False):
        if not weighted:
            for u,v in verts:
                self.adjList[u].append(v)
                if not directed:
                    self.adjList[v].append(u)
        else:
            for u,v,w in verts:
                self.adjList[u].append((v,w))
                if not directed:
                    self.adjList[v].append((u,w))

    def createEdgeMatrix(self, verts, weighted= False):
        edgeu = []
        edgev = []
        edgew = []
        if not weighted:
            for u,v in verts:
                edgeu.append(u)
                edgev.append(v)
        else:
            for u,v,w in verts:
                edgeu.append(u)
                edgev.append(v)
                edgew.append(w)

        self.adjMatrix = [[0 for i in range(len(edgeu))] for k in range(len(edgeu))]

        for i in range(len(edgeu)):
            u = edgeu[i]
            v = edgev[i]
            if not weighted:
                self.adjMatrix[u][v] = 1
            else:
                w = edgew[i]
                self.adjMatrix[u][v] = edgew[i]


        for i in range(len(self.adjMatrix)):
            for j in range(len(self.adjMatrix[i])):
                print(self.adjMatrix[i][j], " ", end= " ")
            print(' ')

    def bfs(self, start):
        queue = [start]
        order = []
        visited = set()
        while queue:
            node = queue.pop(0)
            order.append(node)
            visited.add(node)
            for neighbor in self.adjList[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

        print(order)

    def dfs(self, start):
        visited = set()
        order = []
        def search(node):
            if node not in visited:
                visited.add(node)
                order.append(node)
                for neighbor in self.adjList[node]:
                    search(neighbor)
        search(start)
        print(order)

    def dijkstra(self, start):
        visited = set()
        parentMap = {}
        priorityQueue = []
        nodeCosts = defaultdict(lambda: float('inf'))
        nodeCosts[start] = 0
        heap.heappush(priorityQueue, (0, start))

        while priorityQueue:
            _, node = heap.heappop(priorityQueue)
            visited.add(node)

            for i in range(len(self.adjList[node])):
                adjWeight = []
                adjWeight.append((self.adjList[node][i][0], self.adjList[node][i][1]))
                for neighbor, weight in adjWeight:
                    if neighbor in visited:
                        continue
                    newCost = nodeCosts[node] + weight
                    if nodeCosts[neighbor] > newCost:
                        parentMap[neighbor] = node
                        nodeCosts[neighbor] = newCost
                        heap.heappush(priorityQueue, (newCost, neighbor))

        print(parentMap, nodeCosts)

    def bellmanFord(self, start, grid):
        V = set()
        for u,v,w in grid:
            V.add(u)
        vertCount = len(V)
        dist = [float('inf')] * vertCount
        dist[start] = 0

        for _ in range(vertCount - 1):
            for u,v,w in grid:
                if dist[u] != float('inf') and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
            
        for u,v,w in grid:
            if dist[u] != float('inf') and dist[u] + w < dist[v]:
                print("There's a cycle")
                return

        print(dist)

    def kahn(self, grid):
        V = set()
        for u,v in grid:
            V.add(u)
            V.add(v)
        vertCount = len(V)

        inDegree = [0] * vertCount
        for i in self.adjList:
            for j in self.adjList[i]:
                inDegree[j] += 1

        queue = []
        for i in range(vertCount):
            if inDegree[i] == 0:
                queue.append(i)

        count = 0
        order = []
        while queue:
            u = queue.pop(0)
            order.append(u)
            for i in self.adjList[u]:
                inDegree[i] -= 1
                if inDegree[i] == 0:
                    queue.append(i)

            count += 1

        if count != vertCount:
            print("There's a cycle in this graph")
        else:
            print(order)

    def tarjan(self, grid):
        V = set()
        for u,v in grid:
            V.add(u)
            V.add(v)
        vertCount = len(V)

        disc = [-1] * vertCount
        low = [-1] * vertCount
        stackMember = [False] * vertCount
        st = []

        def helper(u,low,disc,stackMember,st):
            disc[u] = self.time
            low[u] = self.time
            self.time += 1
            stackMember[u] = True
            st.append(u)

            for v in self.adjList[u]:
                if disc[v] == -1:
                    helper(v,low,disc, stackMember, st)
                    low[u] = min(low[u], low[v])
                elif stackMember[v] == True:
                    low[u] = min(low[u], disc[v])

            w = -1
            if low[u] == disc[u]:
                while w != u:
                    w = st.pop()
                    print(w, end= " ")
                    stackMember[w] = False
                print(' ')
                    

        for i in range(vertCount):
            if disc[i] == -1:
                helper(i,low,disc,stackMember, st)
                


g1 = Graph(graph)
g1.createAdjList(graph)
g1.createEdgeMatrix(graph)
g1.bfs(0)
g1.dfs(0)

g2 = Graph(weightedGraph)
g2.createAdjList(weightedGraph, True, True)
g2.createEdgeMatrix(weightedGraph, True)
g2.dijkstra(0)
g2.bellmanFord(0, weightedGraph)

g3 = Graph(graph)
g3.createAdjList(graph, True)
g3.kahn(graph)
g3.dfs(5)
g3.tarjan(graph)

