
graph = [(5,2), (5,0), (4,0), (4,1), (2,3), (3,1)]
weightedGraph = [(0,1,8), (0,4,3), (5,3,1), (2, 1,6), (1,4,3), (3,4,5), (4,5,2)]

def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        queue = collections.deque()
        rows, cols = len(matrix), len(matrix[0])
        
        res = [[float('inf') for _ in range(cols)] for _ in range(rows)]
        
        for i in range(rows):
            for j in range(cols):
                if matrix[i][j] == 0:
                    res[i][j] = 0
                    queue.append((i, j))
                    
        while queue:
            i, j = queue.pop()
            directions = [(0,1),(0,-1),(1,0),(-1,0)]
            
            for dx, dy in directions:
                newX, newY = i+dx, j+dy
                if 0<=newX<rows and 0<=newY<cols and matrix[newX][newY] == 1 and res[newX][newY] > res[i][j]+1:
                    res[newX][newY] = res[i][j]+1
                    queue.appendleft((newX, newY))
        
        return res

from math import sqrt, ceil
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums
        n = len(nums)
        self.m = int(sqrt(n))
        self.small_sums = [0]*(ceil(n/self.m))
        
        for i in range(n):
            self.small_sums[i//self.m] += nums[i]
        

    def update(self, index: int, val: int) -> None:
        change = val - self.nums[index]
        self.small_sums[index//self.m] += change
        self.nums[index] = val

    def sumRange(self, left: int, right: int) -> int:
        result = 0
        s_left = left//self.m
        s_right = right//self.m
        # Add all the blocks
        for i in range(s_left, s_right+1):
            result+= self.small_sums[i]
            
        if left % self.m!= 0:
            for i in range(s_left*self.m, left):
                result -= self.nums[i]
                
        right_end = min(len(self.nums),(s_right+1)*self.m)
        for i in range(right+1, right_end):
            result -= self.nums[i]
            
        return result

def largestTimeFromDigits(self, arr):
    
    for i in itertools.permutations(sorted(arr,reverse=True)):
        if int(str(i[0])+str(i[1])) < 24 and int(str(i[2])+str(i[3])) <59:
            return str(i[0])+str(i[1])+":"+str(i[2])+str(i[3])
    return ""

def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        p1=head
        p2=head
        cnt=0
        while cnt<n:
            p1=p1.next
            cnt+=1

        if not p1: # n equals the length of the linked list
            return head.next

        while p1.next:
            p1=p1.next
            p2=p2.next
        p2.next=p2.next.next

        return head

def splitArraySameAverage(self, A: List[int]) -> bool:
        total = sum(A)
        # The ans does not change if we scale or shift the array
        for i in range(len(A)): A[i] = len(A) * A[i] - total
        # The above ensures that sum(A)=Avg(A)=avg(B)=avg(C)=sum(B)=sum(C)=0
        # So we are looking for a non-empty strict subset of A that sum to 0
        
        A.sort() #help prune the search a bit
        X = set()
        for a in A[:-1]: #excluding last element so it looks for STRICT subset
            X |= { a } | { a + x for x in X if x < 0}
            if 0 in X: return True
        return False

def maxProfit(self, prices: List[int]) -> int:
        if not len(prices):
            return 0
        
        totalProfit = [[0 for i in range(len(prices))] for k in range(len(prices)//2)]
        
        for t in range(1, len(prices)//2):
            maxAmount = float('-inf')
            for d in range(1, len(prices)):
                maxAmount = max(maxAmount, totalProfit[t-1][d-1] - prices[d-1])
                totalProfit[t][d] = max(totalProfit[t][d - 1], maxAmount + prices[d])
                
                
        return totalProfit[-1][-1]


