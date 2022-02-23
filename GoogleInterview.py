import sys
import string
from collections import defaultdict
from collections import deque

class KthLargest:
    def __init__(self, k, nums):
        self.numberList = nums
        self.k = k

    def add(self, val):
        self.numberList.append(val)
        self.numberList.sort()
        return self.numberList[len(self.numberList) - self.k]

def deleteNodeLinkedList():
    node.val = node.next.val
    node.next = node.next.next

def deleteNodeLinkedList_WithVale(val):
    temp = head
    while temp != None:
        while temp.next != None and temp.next.val == val:
            temp.next = temp.next.next

        if head != None and head.val == val:
            head = head.next

    return head

def deleteDuplicatesfrom_SortedLinkedList(head):
    if head == None:
        return None

    while head.next != None and head.val == head.next.val:
        head.next = head.next.next
    
    head.next = self.deleteDuplicatesfrom_SortedLinkedList(head.next)
    return head

def reverseLinkedList(head):
    if head == None:
        return head

    current = head
    prev = current
    prev.next = None

    while current:
        temp = current
        current = current.next
        temp.next = prev
        prev = temp

def findMiddle_LinkedList(head):
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow

def copyLinkedList_wPointer(root):
    current = root
    while current != None:
        new = Node(current.data)
        new.next = current.next
        current.next = new
        current = current.next.next

    current = root
    while current != None:
        current.next.random = current.random.next
        current = current.next.next

    current = root
    duplicate_root = root.next
    while current.next != None:
        temp = current.next
        current.next = current.next.next
        current = temp
    return duplicate_root

def InvertBinaryTree(tree):
    if tree:
        tree.left, tree.right = tree.right, tree.left
        InvertBinaryTree(tree.left)
        InvertBinaryTree(tree.right)

def printPath_Utility(root, path, val):
    if root == None:
        return

    path.append(root.data)
    printPath_Utility(root.left, path, k)
    printPath_Utility(root.right, path, k)

    combinedVal = 0
    for j in range(len(path) -1, -1, -1):
        combinedVal += path[j]
        if combinedVal == val:
            printVector(path, j)
    path.pop(-1)

def printPath(root, val):
    path = []
    printPath_Utility(root, path, val)

def longest_Substring_w_KDistinct(str, k):
    if str == None:
        return 0

    start, maxVal = 0,0
    seen = set()
    for end in range(len(str)):
        seen.add([str[end]])
        while len(seen) > k:
            if str[start] in seen:
                seen.remove(str[start])
                start += 1
                maxVal = max(maxVal, end - start + 1)

    return maxVal

def longestUniqueSubstring(string):
    last_idx = {}
    max_len = 0
    start_idx = 0

    for i in range(0, len(string)):
        if string[i] in last_idx:
            start_idx = max(start_idx, last_idx[string[i]] + 1)
        max_len = max(max_len, i - start_idx + 1)
        last_idx[string[i]] = i
    return max_len

def isSubsetSum(arr, n, sum):
    if sum == 0:
        return True
    if n == 0 and sum != 0:
        return False

    if arr[n-1] > sum:
        return isSubsetSum(arr, n-1, sum)
    return isSubsetSum(arr, n-1, sum - arr[n-1]) or isSubsetSum(arr, n-1, sum)

def findPartion(arr, n):
    sum = 0
    for i in range(0,n):
        sum += arr[i]
    if sum%2 != 0:
        return False
    
    return isSubsetSum(arr, n, sum/2)

def isNumeric(s):
    s.strip()
    if s[-1] == ".":
        return print(False)
    try:
        s = float(s)
        return print(True)
    except:
        return print(False)

def printParenthesis(str, n):
    if n > 0:
        _printParenthesis(str, 0, n, 0, 0)
    return

def _printParenthesis(str, pos, n, open, close):
    if close == n:
        for i in str:
            print(i, end= "")
            print()
            return
        else:
            if open > close:
                str[pos] = '}'
                _printParenthesis(str, pos + 1, n, open, close + 1)
            if open < n:
                str[pos] = '{'
                _printParenthesis(str, pos + 1, n, open + 1, close)

def findFirstOccurence(A, x):
    left, right = 0, len(A) - 1
    result = -1

    while left <= right:
        mid = (left + right)//2
        if x == A[mid]:
            result = mid
            right = mid - 1
        elif x < A[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return result

def findLastOccurence(A, x):
    left, right = 0, len(A) - 1
    result = -1
    while left <= right:
        mid = (left + right)//2
        if X == A[mid]:
            result = mid
            left = mid + 1
        elif x < A[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return result

def mergeIntervals(arr):
    arr.sort(key= lambda x: x[0])
    merged = []
    s = -sys.maxsize
    max = -sys.maxsize
    for i in range(len(arr)):
        a = arr[i]
        if a[0] > max:
            if i != 0:
                merged.append([s,max])
                max = a[1]
                s = a[0]
            else:
                if a[i] >= max:
                    max = a[1]

    if max != -sys.maxsize and [s, max] not in merged:
        merged.append([s, max])

def heightBinaryTree(root):
    if root != None:
        return max(1 + heightBinaryTree(root.left), 1 + heightBinaryTree(root.right))
    else:
        return -1

def get_level(self):
    level = 0
    p = self.parent
    while p:
        level += 1
        p = p.parent

    return level

def insertLinkedNode(head, data, position):
    start = head
    if position == 0:
        return Node(data, head)
    while position > 1:
        head = head.next
        position -= 1
    
    head.next = Node(data, head.next)
    return start

def fibonacci(n):
    if n < 0:
        return -1
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return fibonacci(n -1) + fibonacci(n-2)

def mostCommonWord_ExcludeBanned(self, paragraph, banned):
    freqCount = {}
    bannedWords = {}
    currentVal = 0
    outputWord = ""

    for c in string.punctuation:
        paragraph = paragraph.replace(c, " ")

    paragraph = ((paragraph).lower()).split()
    for i in banned:
        if i not in bannedWords:
            bannedWords[i] = 1

    for i in paragraph:
        if i not in bannedWords:
            if i not in freqCount:
                freqCount[i] = 1
            else:
                freqCount[i] += 1

    for k, v in freqCount.items():
        if v > currentVal:
            currentVal = v
            outputWord = k

    return outputWord

def findValue_BinaryTree_betweenValues(self, root, low, hight):
    self.ans = 0
    self.low = low
    self.high = hight
    self.rangesum(root)
    return self.ans

    def rangesum(self, node):
        if node != None:
            return

        if node.val < self.low:
            self.rangesum(node.right)
        elif node.val > self.high:
            self.rangesum(node.left)
        else:
            self.ans += node.val
            self.rangesum(node.left)
            self.rangesum(node.right)

def numTilePossibilities(self, tiles):
    n = len(tiles)
    visited = [False for _ in range(n)]
    res = [0]
    used = set()

    def dfs(ans):
        if len(ans) > 0:
            used.add("".join(ans))
        if len(ans) >= n:
            return
        for i in range(n):
            if visited[i]:
                continue
            visited[i] = True
            dfs(ans + tiles[i])
            visited[i] = False

    dfs([])
    return len(used)

def gardenNoAdj(self, n, paths):
    g = defaultdict(set)
    for x,y in paths:
        g[x].add(y)
        g[y].add(x)

    garden = [0] * (n+1)
    for i in range(1, n+1):
        plants = {1,2,3,4}

    for idx in g[i]:
        if garden[idx] in plants:
            plants.remove(garden[idx])
    garden[i] = plants.pop()
    return garden[1:]

def width_BinaryTree(self, root):
    queue = [(root, 0, 0)]
    curr_depth = left = ans = 0
    for node, depth, pos in queue:
        if node:
            queue.append((node.left, depth +1, pos *2))
            queue.append((node.right, depth +1, pos *2 + 1))
            if curr_depth != depth:
                curr_depth = depth
                left = pos
                ans = max(pos - left + 1, ans)

    return ans

def twoSums(self, nums, target):
    numberDict = {}
    for i in range(len(nums) - 1):
        if nums[i] not in numberDict:
            numberDict[target - nums[i]] = i

    for idx, val in enumerate(nums):
        if val in numberDict and numberDict[val] != idx:
            correct = numberDict[val]
            return [correct, idx]

def sudoku(board):
    cacheRows = [[] for i in range(9)]
    cacheCols = [[] for i in range(9)]
    cacheSquares = [[] for i in range(9)]

    for rowIdx, row in enumerate(board):
        for valIdx, val in enumerate(row):
            if val == ".":
                continue
            squareIndex = rowIdx//3 * 3 + valIdx//3

            for cache in [cacheRows[rowIdx], cacheCols[valIdx], cacheSquares[squareIndex]]:
                if val in cache:
                    return False
                cache.append(val)
    return True

def dfs_Graph_CopyOld2New(old):
    d = {}
    if old not in d:
        new = d[old] = Node(old.val, None)
        new.neighbors = list(map(dfs_Graph_CopyOld2New, old.neighbors))
        return d[old]
    
    return node and dfs_Graph_CopyOld2New(node)

def canWinNim(self, n):
    dp = [False, False, False, True]
    for i in range (0, n-4):
        j = 1
        while j <= 3:
            if dp[-j] == True:
                break
            j += 1
        
        if j == 4:
            dp.append(True)
        else:
            dp.append(False)

    return not dp[n -1]

    #Alternate Solution return n%4

def djkstra(graph, start, end):
    unVisited = graph
    distance2Origin = {}
    priorLocation = {}
    finalRoute = []

    for i in unVisited:
        distance2Origin[i] = sys.maxsize
    distance2Origin[start] = 0

    while unVisited:
        minDistancedNode = None
        for i in unVisited:
            if minDistancedNode == None:
                minDistancedNode = i
            if distance2Origin[minDistancedNode] > distance2Origin[i]:
                minDistancedNode = i

        for node, distance in graph[minDistancedNode].items():
            if distance2Origin[node] > distance2Origin[minDistancedNode] + distance:
                distance2Origin[node] = distance2Origin[minDistancedNode] + distance
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

def tarjan(self, connections):
    d = defaultdict(list)

    for c1, c2 in connections:
        d[c1].append(c2)
        d[c2].append(c1)

    idx = defaultdict(lambda: float('inf'))
    low = defaultdict(lambda: float('inf'))
    ans = []
    self.time = 0

    def _dfsTarjin(i, prev):
        idx[i] = self.time
        low[i] = self.time
        self.time += 1

        for j in d[i]:
            if j == prev:
                continue
            if idx[j] == float('inf'):
                _dfsTarjin(j,i)
            if idx[i] < low[j]:
                ans.append([i,j])

            low[i] = min(low[i], low[j])

    _dfsTarjin(0, None)
    return ans

class TreeNode:
    def __init__(self, v):
        self.val = v
        self.children = {}
        self.endhere = False

class Trie:
    def __init__(self):
        self.root = TreeNode(None)

    def insert(self, word):
        parent = self.root
        for i, char in enumerate(word):
            if char not in parent.children:
                parent.children[char] = TreeNode(char)
            parent = parent.children[char]
            if i == len(word) - 1:
                parent.endhere = True

    def search(self, word):
        parent = self.root
        for char in word:
            if char not in parent.children:
                return False
            parent = parent.children[char]
        return parent.endhere

    def startsWith(self, prefix):
        parent = self.root
        for char in prefix:
            if char not in parent.children:
                return False
            parent = parent.children[char]
        return True

def isRectangleOverlap(self, rec1, rec2):
        xl = min(rec1[2], rec2[2]) - max(rec1[0], rec2[0])
        yl = min(rec1[-1], rec2[-1]) - max(rec1[1], rec2[1])
        return xl > 0 and yl > 0

def updateMatrix(self, matrix):
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                matrix[i][j] = self.bfs(matrix, i, j)
    return matrix
    
def bfs(self, matrix, x, y):
    queue = deque([(x, y, 0)])
    visited = set()
    
    while queue:
        x, y, steps = queue.popleft()
        
        if len(matrix) > x > -1 and len(matrix[0]) > y > -1 and (x, y) not in visited:
            visited.add((x, y))

            if matrix[x][y] == 0:
                return steps

            queue.append((x + 1, y, steps + 1))
            queue.append((x - 1, y, steps + 1))
            queue.append((x, y + 1, steps + 1))
            queue.append((x, y - 1, steps + 1))

    return matrix[x][y]

def findTarget(self, root, k):
    self.tempList = []
    sumHolder = {}
    def inorder(root):
        
        if root:
            inorder(root.left)
            self.tempList.append(root.val)
            inorder(root.right)
        return
    inorder(root)
    for index, number in enumerate(self.tempList):
        
        sumHolder[number] = index

    for index, number in enumerate(self.tempList):
        
        if k - number in sumHolder and index != sumHolder[k-number]:
            return True
    return False

graph = {
    'A' : ['B','C'],
    'B' : ['D', 'E'],
    'C' : ['F'],
    'D' : [],
    'E' : ['F'],
    'F' : []
}

visited = set() # Set to keep track of visited nodes.

def dfs(visited, graph, node):
    if node not in visited:
        print (node)
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

# Driver Code
dfs(visited, graph, 'A')

graph = {
  'A' : ['B','C'],
  'B' : ['D', 'E'],
  'C' : ['F'],
  'D' : [],
  'E' : ['F'],
  'F' : []
}

visited = [] # List to keep track of visited nodes.
queue = []     #Initialize a queue

def bfs(visited, graph, node):
  visited.append(node)
  queue.append(node)

  while queue:
    s = queue.pop(0) 
    print (s, end = " ") 

    for neighbour in graph[s]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)

# Driver Code
bfs(visited, graph, 'A')

def all_subset(given_arr):
    subset = [-1] * len(given_arr)
    helper(given_arr, subset, 0)

def helper(given_arr, subset, i):
    if i == len(given_arr):
        print_set(subset)
    else:
        subset[i] = -1
        helper(given_arr, subset, i + 1)
        subset[i] = given_arr[i]
        helper(given_arr, subset, i + 1)

def print_set(subset):
    finalSub = []
    for i in subset:
        if i != -1:
            finalSub.append(i)

    print(finalSub)
            

all_subset([0,8,5,8,2])

class Solution:
    def numUniqueEmails(self, emails: List[str]) -> int:
        count = 0
        newList = []
        for i in emails:
            newList.append(i.split("@"))
            
        for i in range(len(emails)):
            for idx, val in enumerate(newList[i][0]):
                if val == "+":
                    newList[i][0] = newList[i][0][:idx]
                    break
                    
        for i in range(len(emails)):
            newList[i][0] = newList[i][0].replace('.', "")
            
        finalList = []
            
        for i in range(len(emails)):
            for j in newList[i][1]:
                if j.count('.') == 1:
                    finalList.append(newList[i])
                    
        newSet = []
                    
                    
        for i in finalList:
            if i not in newSet:
                newSet.append(i)
            
                
                
        return len(newSet)