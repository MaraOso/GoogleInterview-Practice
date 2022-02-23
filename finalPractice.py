from collections import defaultdict
import heapq as heap

class Graph:
    def __init__(self, size):
        self.size = size
        self.adjMatrix = defaultdict(list)
        self.time = 0

    def addPositions(self, array, weighted= False):
        if not weighted:
            for u,v in array:
                self.adjMatrix[u].append(v)
                #self.adjMatrix[v].append(u)
        else:
            for u,v,w in array:
                self.adjMatrix[u].append([v,w])
                #self.adjMatrix[v].append([u,w])

    def bfs(self, start):
        queue = [start]
        visited = set()
        while queue:
            node = queue.pop(0)
            print(node, end= " ")
            visited.add(node)
            for neighbor in self.adjMatrix[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
                    visited.add(neighbor)

    def dfs(self, start):
        visited = set()
        def search(node):
            visited.add(node)
            print(node, end= " ")
            for neighbor in self.adjMatrix[node]:
                if neighbor not in visited:
                    search(neighbor)

        search(start)


    def dijkstra(self, start):
        priorityQueue = []
        dist2 = [float('inf')] * self.size
        dist2[start] = 0
        parents = {}
        heap.heappush(priorityQueue, (0, start))
        visited = set()

        while priorityQueue:
            _, node = heap.heappop(priorityQueue)
            visited.add(node)

            for i in range(len(self.adjMatrix[node])):
                adjArray = []
                adjArray.append((self.adjMatrix[node][i][0], self.adjMatrix[node][i][1]))

                for neighbor, distance in adjArray:
                    if neighbor in visited:
                        continue
                    newCost = dist2[node] + distance
                    if dist2[neighbor] > newCost:
                        dist2[neighbor] = newCost
                        heap.heappush(priorityQueue, (newCost, neighbor))
                        parents[neighbor] = node

        print(parents, dist2)

    def bellmanFord(self, start, graph):
        dist2 = [float('inf')] * self.size
        dist2[start] = 0
        vSet = set()
        for u,v,w in graph:
            vSet.add(u)

        for _ in range(len(vSet) - 1):
            for u,v,w in graph:
                if dist2[u] != float('inf') and dist2[u] + w < dist2[v]:
                    dist2[v] = dist2[u] + w

        for u,v,w in graph:
            if dist2[u] != float('inf') and dist2[u] + w < dist2[v]:
                print("There's a cycle")
                return

        print(dist2)

    def kahn(self):
        inDegree = [0] * self.size

        for i in self.adjMatrix:
            for j in self.adjMatrix[i]:
                inDegree[j] += 1

        queue = []
        order = []
        count = 0
        for i in range(self.size):
            if inDegree[i] == 0:
                queue.append(i)

        while queue:
            node = queue.pop(0)
            order.append(node)
            for i in self.adjMatrix[node]:
                inDegree[i] -= 1
                if inDegree[i] == 0:
                    queue.append(i)
            count += 1

        if count != self.size:
            print("Theres a cycle")
        else:
            print(order)

    def tarjan(self):
        discTime = [-1] * self.size
        lowestTime = [-1] * self.size
        stackMembers = [False] * self.size
        stack = []

        def tarjanDFS(u, discTime, lowestTime, stackMembers, stack):
            discTime[u] = self.time
            lowestTime[u] = self.time
            self.time += 1
            stackMembers[u] = True
            stack.append(u)

            for v in self.adjMatrix[u]:
                if discTime[v] == -1:
                    tarjanDFS(v, discTime, lowestTime, stackMembers, stack)
                    lowestTime[u] = min(lowestTime[u], lowestTime[v])
                elif stackMembers[v] == True:
                    lowestTime[u] = min(lowestTime[u], discTime[v])

            w = -1
            if discTime[u] == lowestTime[u]:
                while w != u:
                    w = stack.pop()
                    print(w, end= " ")
                    stackMembers[w] = False
                print(' ')

        for i in range(self.size):
            if discTime[i] == -1:
                tarjanDFS(i, discTime, lowestTime, stackMembers, stack)

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

        for i in self.adjMatrix:
            for j in self.adjMatrix[i]:
                union(data, i, j)

        for i in range(self.size):
            print("item", i, "connected compenent", find(data, i))

        print(connected(data, 0,3))

def maxProfit(prices, k):
    profit = [0] * k       # profit without position (aka not holding stocks, all cash)
    balance = [-1000] * k  # balance with open position (aka has 1 stock + some cash)
    for price in prices:
        for t in range(k):
            balance[t] = max(balance[t], -price + (profit[t - 1] if t else 0))
            profit[t] = max(profit[t], balance[t] + price)
    print(profit[-1] if k else 0)

maxProfit([1,8,3,7], 2)

