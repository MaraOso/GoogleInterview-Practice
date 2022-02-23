def fib(n, memo={}):
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1
    memo[n] = fib(n-1, memo) + fib(n - 2, memo)
    return memo[n]

def gridTraveler(n, m, memo= {}):
    key = str(n) + ',' + str(m)
    if key in memo: 
        return memo[key]
    if n == 1 and m == 1:
        return 1
    if n == 0 or m == 0:
        return 0
    memo[key] = gridTraveler(n-1,m, memo) + gridTraveler(n,m-1, memo)
    return memo[key]

#Make it work (Recursive) --- View as trees (Graph) --- Impliment Trees as recursive --- Leaves are base case --- test it
#Make it efficient --- add memo object (key with return value) --- shared --- add memo base case --- store memo always

def canSum(target, nums, memo={}):
        if target in memo:
            return memo[target]
        if target == 0:
            return True
        if target < 0:
            return False
        for i in nums:
            if canSum(target - i, nums, memo) == True:
                memo[target] = True
                return True

        memo[target] = False
        return False

def howSum(targetSum, numbers, memo={}):
    if targetSum in memo:
        return memo[targetSum]
    if targetSum == 0:
        return []
    if targetSum < 0:
        return None
    for i in numbers:
        remainder = targetSum - i
        remainderResult = howSum(remainder, numbers, memo)
        if remainderResult != None:
            memo[targetSum] =  remainderResult + [i]
            return memo[targetSum]

    memo[targetSum] = None
    return None

def bestSum(targetSum, numbers, memo={}):
    if targetSum in memo:
        return memo[targetSum]
    if targetSum == 0:
        return []
    if targetSum < 0:
        return None

    shortestArr = None

    for i in numbers:
        remainder = targetSum - i
        result = bestSum(remainder, numbers, memo)
        if result != None:
            combination = result + [i]
            if shortestArr == None or len(shortestArr) > len(combination):
                shortestArr = combination

    memo[targetSum] = shortestArr
    return shortestArr


def canConstruct(target, wordBank, memo= {}):
    if target in memo:
        return memo[target]
    if target == '':
        return True
    for i in wordBank:
        if i == target[:len(i)]:
            found = canConstruct((target.replace(i, "")), wordBank, memo)
            if found:
                memo[target] = True
                return True

    memo[target] = False
    return False
    
def countConstruct(target, wordBank, memo= {}):
    if target in memo:
        return memo[target]

    if target == "":
        return 1

    totalCount = 0
    for i in wordBank:
        if i == target[:len(i)]:
            foundNum = countConstruct(target.replace(i, ""), wordBank, memo)
            totalCount += foundNum

    memo[target] = totalCount
    return totalCount

def allConstruct(target, wordBank, memo={}):
    collection = []
    if target == "":
        return [[]]

    for i in wordBank:
        if i == target[:len(i)]:
            foundWord = allConstruct(target.replace(i, ""), wordBank, memo)
            Words = [[i] + x for x in foundWord]

            collection.extend(Words)

    return collection
    
def tfib(n):
    array = [0 for i in range(n + 1)]
    array[1] = 1
    for i in range(2,n + 1):
        array[i] = array[i] + array[i -1] + array[i - 2]

    return array[n]

def tgridTraveler(n,m):
    array = [[0 for i in range(n + 1)] for j in range(m + 1)]
    array[1][1] = 1
    for i in range(m + 1):
        for j in range(n + 1):
            current = array[i][j]
            if i + 1 <= m:
                array[i+ 1][j] += current
            if j + 1 <= n:
                array[i][j + 1] += current

    return array[n][m]

#print(tgridTraveler(18,18))

def tcanSum(target, arr):
    array = [False for i in range(target + 1)]

    array[0] = True

    for i in range(target + 1):
        if array[i] == True:
            for j in arr:
                if i + j < target + 1:
                    array[i + j] = True

    return array[target]

#print(tcanSum(14, [6,12,5]))

def thowSum(target, arr):
    array = [None for i in range(target + 1)]
    array[0] = []

    for i in range(target + 1):
        if array[i] != None:
            for j in arr:
                if i + j < target + 1:
                    array[i + j] = array[i] + [j]

    return array[target]

print(thowSum(24, [12,6,5]))

