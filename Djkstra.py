import sys

graph = {'a': {'b': 10, 'c': 13}, 'b': {'c': 1, 'd': 2}, 'c': {'b': 4, 'd':8, 'e': 12, 'f': 22}, 'd': {'e': 7}, 'e': {'d': 9}, 'f': {'e': 21}}
graph2 = {1:2, 2:3, 3:4, 4:5}

def djkstra(graph, start, end):
    unVisited = graph
    distances2Origin = {}
    priorLocation = {}
    finalRoute = []

    for i in unVisited:
        distances2Origin[i] = sys.maxsize
    distances2Origin[start] = 0

    while unVisited:
        minDistancedNode = None
        for i in unVisited:
            if minDistancedNode == None:
                minDistancedNode = i
            if distances2Origin[minDistancedNode] > distances2Origin[i]:
                minDistancedNode = i

        for node, distance in graph[minDistancedNode].items():
            if distances2Origin[node] > distances2Origin[minDistancedNode] + distance:
                distances2Origin[node] = distances2Origin[minDistancedNode] + distance
                priorLocation[node] = minDistancedNode
        unVisited.pop(minDistancedNode)

    currentLocation = end
    while currentLocation != start:
        try:
            finalRoute.insert(0, currentLocation)
            currentLocation = priorLocation[currentLocation]
        except KeyError:
            return -1

    finalRoute.insert(0, start)
    print(finalRoute, distances2Origin[end])

djkstra(graph, "a", "f")
#djkstra(graph2, 1, 5)


