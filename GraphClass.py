graph = [(5,2), (5,0), (4,0), (4,1), (2,3), (3,1)]
weightedGraph = [(0,1,8), (0,4,3), (5,3,1), (2, 1,6), (1,4,3), (3,4,5), (4,5,2), (0,2,7)]
floydWeight = [(0,1,16), (0,2,13), (1,2,10), (1,3,12), (2,1,4), (2,4,14), (3,2,9), (3,5,20), (4,3,7), (4,5,4)]

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
            for u,v,_ in verts:
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

    def primsMST(self, grid, start):
        V = set()
        for u,v,_ in grid:
            V.add(u)
            V.add(v)
        vertCount = len(V)

        selected = [False] * vertCount
        selected[start] = True
        no_edge = 0
        while no_edge < vertCount - 1:
            minimum = float('inf')
            x= 0
            y= 0
            for i in range(vertCount):
                if selected[i]:
                    for j in range(vertCount):
                        if not selected[j] and self.adjMatrix[i][j]:
                            if minimum > self.adjMatrix[i][j]:
                                minimum = self.adjMatrix[i][j]
                                x=i
                                y=j
            print(str(x) + '-' + str(y) + "-" + str(self.adjMatrix[x][y]))
            selected[y] = True
            no_edge += 1

    def fordFulkerson(self, grid, source, sink):
        V = set()
        for u,v,_ in grid:
            V.add(u)
            V.add(v)
        vertCount = len(V)
        parent = [-1] * vertCount
        max_flow = 0
        
        def fordFulkerson_BFS(s, t, parent):
            #visited = set()
            visited = [False] * vertCount
            queue = []
            queue.append(s)
            #visited.add(s)
            visited[s] = True
            while queue:
                u = queue.pop(0)
                for ind, val in enumerate(self.adjMatrix[u]):
                    if visited[ind] == False and val > 0:
                    #if ind not in visited and val > 0:
                        if ind == t:
                            visited[ind] = True
                            #visited.add(ind)
                            return True
                    queue.append(ind)
                    #visited.add(ind)
                    visited[ind] = True
                    parent[ind] = u
            return False

        while fordFulkerson_BFS(source, sink, parent):
            path_flow = float('inf')
            s= sink
            while s != source:
                path_flow = min(path_flow, self.adjMatrix[parent[s]][s])
                s = parent[s]

            max_flow += path_flow
            v = sink
            while v != source:
                u = parent[v]
                self.adjMatrix[u][v] -= path_flow
                self.adjMatrix[v][u] += path_flow
                v = parent[v]

        return max_flow


    def edmondsKarp():
        pass

    def unionFind(self):
        parent = [-1] * self.size

        def find_parent(parent, i):
            if parent[i] == -1:
                return i
            if parent[i] != -1:
                return find_parent(parent, parent[i])

        def union(parent, x, y):
            parent[x] = y

        for i in self.adjList:
            for j in self.adjList[i]:
                x = find_parent(parent, i)
                y = find_parent(parent, j)
                if x == y:
                    return True
                union(parent, x, y)
        return False
        


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

g2.primsMST(weightedGraph, 0)

g4 = Graph(floydWeight)
g4.createEdgeMatrix(floydWeight, True)
print(g4.fordFulkerson(weightedGraph, 0, 4))


