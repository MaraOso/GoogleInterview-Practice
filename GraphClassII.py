from collections import defaultdict
import heapq as heap 

class Graph:
    def __init__(self, size):
        self.adjList = defaultdict(list)
        self.adjMatrix = []
        for _ in range(size):
            self.adjMatrix.append([0 for i in range(size)]) #float('inf') or 0
        self.size = size
        self.time = 0

    def add_edge(self, v1, v2):
        self.adjMatrix[v1][v2] = 1
        self.adjMatrix[v2][v1] = 1
        self.adjList[v1].append(v2)
        self.adjList[v2].append(v1)

    def add_edges(self, grid, weighted= False, directed = False):
        if not weighted:
            for u,v in grid:
                self.adjMatrix[u][v] = 1
                self.adjList[u].append(v)
                if not directed:
                    self.adjMatrix[v][u] = 1
                    self.adjList[v].append(u)
        else:
            for u,v,w in grid:
                self.adjMatrix[u][v] = w
                edgeVal = (v,w)
                self.adjList[u].append(edgeVal)
                if not directed:
                    edgeVal2 = (u,w)
                    self.adjList[v].append(edgeVal2)
                    self.adjMatrix[v][u] = w

    def remove_edge(self, v1, v2):
        if self.adjMatrix[v1][v2] == 0:
            return
        self.adjMatrix[v1][v2] = 0
        self.adjMatrix[v2][v1] = 0

    def __len__(self):
        return self.size

    def print_matrix(self):
        for row in self.adjMatrix:
            for val in row:
                print(val, " ", end= " ")
            print(" ")

    def print_list(self):
        print(self.adjList)

    def bfs(self, start, Matrix= False):
        visited = set()
        queue = [start]
        visited.add(start)
        order = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            visited.add(current)
            if not Matrix:
                for neighbor in self.adjList[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            else:
                for neighbor in range(self.size):
                    if self.adjMatrix[current][neighbor] == 1 and neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)

        print(order)

    def dfs(self, start, Matrix= False):
        visited = set()
        order = []
        def search(node):
            if node not in visited:
                visited.add(node)
                order.append(node)
                if not Matrix:
                    for neighbor in self.adjList[node]:
                        if neighbor not in visited:
                            search(neighbor)
                else:
                    for neighbor in range(self.size):
                        if self.adjMatrix[node][neighbor] == 1 and neighbor not in visited:
                            search(neighbor)
        search(start)
        print(order)

    def dijkstra(self, start, Matrix= False):
        visited = set()
        dist2Start = defaultdict(lambda: float('inf'))
        dist2Start[start] = 0
        priorityQueue = []
        heap.heappush(priorityQueue, (0, start))
        order = {}

        while priorityQueue:
            _, current = heap.heappop(priorityQueue)
            visited.add(current)
            if not Matrix:
                for i in range(len(self.adjList[current])):
                    adjWeight = []
                    adjWeight.append((self.adjList[current][i][0], self.adjList[current][i][1]))

                    for neighbor, weight in adjWeight:
                        if neighbor in visited:
                            continue
                        newCost = dist2Start[current] + weight
                        if dist2Start[neighbor] > newCost:
                            dist2Start[neighbor] = newCost
                            heap.heappush(priorityQueue, (newCost, neighbor))
                            order[neighbor] = current
            else:
                for neighbor in range(self.size):
                    if self.adjMatrix[current][neighbor] != 0 and neighbor not in visited:
                        newCost = dist2Start[current] + self.adjMatrix[current][neighbor]
                        if dist2Start[neighbor] > newCost:
                            dist2Start[neighbor] = newCost
                            heap.heappush(priorityQueue, (newCost, neighbor))
                            order[neighbor] = current

        print(dist2Start, order)

    def bellmanFord(self, start, grid):
        dist2Start = defaultdict(lambda: float('inf'))
        dist2Start[start] = 0
        Vset = set()
        for u,v,_ in grid:
            Vset.add(u)

        for _ in range(len(Vset) - 1):
            for u,v,w in grid:
                if dist2Start[u] != float('inf') and dist2Start[u] + w < dist2Start[v]:
                    dist2Start[v] = dist2Start[u] + w

        for u,v,w in grid:
                if dist2Start[u] != float('inf') and dist2Start[u] + w < dist2Start[v]:
                    print("There's a Negative Cycle")
                    return
                
        print(dist2Start)


    def floydWarshall(self):
        dist2All = list(map(lambda i: list(map(lambda j: j,i)), self.adjMatrix))

        for k in range(self.size):
            for i in range(self.size):
                for j in range(self.size):
                    dist2All[i][j] = min(dist2All[i][j], dist2All[i][k] + dist2All[k][j])

        for i in range(self.size):
            for j in range(self.size):
                if dist2All[i][j] == 'inf':
                    print('Inf', end= " ")
                else:
                    print(dist2All[i][j], end= " ")
            print(' ')

    def kahn(self):
        inDegree = [0] * self.size
        for i in self.adjList:
            for j in self.adjList[i]:
                inDegree[j] += 1


        queue = []
        for i in range(self.size):
            if inDegree[i] == 0:
                queue.append(i)

        count = 0
        order = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for neighbor in self.adjList[current]:
                inDegree[neighbor] -= 1
                if inDegree[neighbor] == 0:
                    queue.append(neighbor)

            count += 1

        if count != self.size:
            print("There's a cycle")
        else:
            print(order)

    def tarjan(self):
        discTime = [-1] * self.size
        lowestTime = [-1] * self.size
        stackMembers = [False] * self.size
        stack = []

        def tarjanDFS(u, lowestTime, discTime, stackMembers, stack):
            discTime[u] = self.time
            lowestTime[u] = self.time
            self.time += 1
            stackMembers[u] = True
            stack.append(u)

            for v in self.adjList[u]:
                if discTime[v] == -1:
                    tarjanDFS(v, lowestTime, discTime, stackMembers, stack)
                    lowestTime[u] = min(lowestTime[u], lowestTime[v])
                elif stackMembers[v] == True:
                    lowestTime[u] = min(lowestTime[u], discTime[v])

            w = -1
            if lowestTime[u] == discTime[u]:
                while w != u:
                    w = stack.pop()
                    print(w, end= " ")
                    stackMembers[w] = False
                print(" ")

        for i in range(self.size):
            if discTime[i] == -1:
                tarjanDFS(i, lowestTime, discTime, stackMembers, stack)

    def prims(self, start):
        def printMST(parent):
            for i in range(1, self.size):
                print(parent[i], "-", i, '\t', self.adjMatrix[i][parent[i]])

        def findMinDistance(key, mstSet):
            min = float('inf')
            min_index = -1
            for v in range(self.size):
                if key[v] < min and mstSet[v] == False:
                    min = key[v]
                    min_index = v
            return min_index

        key = [float('inf')] * self.size
        parent = [None] * self.size
        key[start] = 0
        mstSet = [False] * self.size
        parent[start] = -1

        for _ in range(self.size):
            u = findMinDistance(key, mstSet)
            mstSet[u] = True
            for v in range(self.size):
                if self.adjMatrix[u][v] > 0 and mstSet[v] == False and key[v] > self.adjMatrix[u][v]:
                    key[v] = self.adjMatrix[u][v]
                    parent[v] = u

        printMST(parent)

    def edmondKarp(self, s,t):
        F = [[0] * self.size for i in range(self.size)]

        def edmondKarpBFS(F,s,t):
            queue = [s]
            paths = {s:[]}
            if s == t:
                return paths[s]
            while queue:
                u = queue.pop(0)
                for v in range(len(self.adjMatrix)):
                    if self.adjMatrix[u][v] - F[u][v] > 0 and v not in paths:
                        paths[v] = paths[u] + [(u,v)]
                        print(paths)
                        if v == t:
                            return paths[v]
                        queue.append(v)
            return None

        path = edmondKarpBFS(F, s,t)
        while path:
            flow = min(self.adjMatrix[u][v] - F[u][v] for u,v in path)
            for u,v in path:
                F[u][v] += flow
                F[v][u] -= flow
            path = edmondKarpBFS(F, s,t)
        return sum(F[s][i] for i in range(self.size))

    def unionFind(self):
        data = [i for i in range(self.size)]
        def find(data, i):
            if i != data[i]:
                data[i] = find(data, data[i])
            return data[i]

        def union(data, i, j):
            pi, pj = find(data, i), find(data, j)
            if pi != pj:
                data[pi] = pj

        def connected(data, i, j):
            return find(data, i) == find(data,j)

        for i in self.adjList:
            for j in self.adjList[i]:
                union(data, i,j)

        for i in range(self.size):
            print('item', i, '-> component', find(data, i))

        print(connected(data, 3,1))