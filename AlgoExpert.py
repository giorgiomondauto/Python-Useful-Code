#Find closest value to BST
# https://www.algoexpert.io/questions/Find%20Closest%20Value%20In%20BST
# Theory page 1 of notebook

# Function1
def findClosestValueInBst(tree, target):
    '''
    '''
	return findClosestValueInBstHelper(tree, target, tree.value)

def findClosestValueInBstHelper(tree, target, closest):
	
	if tree is None:
		return closest
	if abs(target - closest) > abs(target - tree.value):
		closest = tree.value
	
	if target < tree.value:
		return findClosestValueInBstHelper(tree.left, target, closest)
	
	elif target> tree.value:
		return findClosestValueInBstHelper(tree.right, target, closest)
	else:
		
		return closest



# This is the class of the input tree. Do not edit.
class BST:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Fibonacci
# function 2
def getNthFib(n):
    # Write your code here.
    n0 = 0
    n1 = 1
    fibonacci = [n0,n1]
    if n == 0:
        return fibonacci[0]
    elif n == 1:
        return  fibonacci[1]
    else:
        i = 2
        while i<=n-1:
            nth = n1+n0
            n0 = n1
            n1 = nth
            fibonacci.append(nth)
            i+=1

    return fibonacci[-1]

# or algoexpert solution
def getNthFib(n):
    # Write your code here.
    if n == 1:
		return 0
	elif n ==2:
		return 1
	else:
		return getNthFib(n-1)+getNthFib(n-2)

	
# function 3
#take an array and return its product sum
# input: values = [1,[2,[3]]]
# output: 23
def productSum(array, multiplier = 1):
	sum = 0
	for element in array:
		if type(element) is list:
			sum+= productSum(element, multiplier+1)
		else:
			sum +=element
	return sum*multiplier


# function 4
#INSERTION SORT Method
# best: O(n) time | O(1) space
# average: O(n^2) time | O(1) space 
# worst: O(n^2) time | O(1) space
def insertionSort(array):
    # Write your code here.
    for j in range(1,len(array)):
        while j>0 and array[j-1]>array[j]:
            array = swap(j-1,j,array)
            j-=1
            
    return array

def swap(i,j,array):
    array[i],array[j] = array[j],array[i]
    return array

values = [1,5,8,33,4]
print(insertionSort(values))

# Function 5
# BranchSums

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
		
def branchSums(root):
	sums = []
	calculateBranchSums(root, 0 , sums)
	return sums
	

def calculateBranchSums(node,runningsum,sums):
	
	if node is None:
		return sums
	
	newrunningsums = runningsum + node.value
	if node.left is None and node.right is None:
		sums.append(newrunningsums)
		return sums
	
	calculateBranchSums(node.left,newrunningsums,sums)
	calculateBranchSums(node.right,newrunningsums,sums)

# Function 6
# Selection Sort
def selectionSort(array):
    currentIdx = 0
    while currentIdx < len(array)-1:
        smallestIdx = currentIdx
        for i in range(currentIdx + 1, len(array)):
            if array[smallestIdx] > array[i]:
                smallestIdx = i
        swap(currentIdx, smallestIdx, array)
        currentIdx +=1
    return array


def swap(i,j,array):
    array[i],array[j] = array[j],array[i]
    
    
print(selectionSort([1,5,4,77,5555]))
	
# Function 7
# ThreeNumberSum problem
def threeNumberSum(array, targetSum):
    triples = [[array[i],array[j],array[z]] 
			  for i in range(len(array))
			  for j in range(i+1,len(array)) 
			  for z in range(j+1, len(array)) 
			  if array[i]+array[j]+array[z] == targetSum]
	for i in triples:
		i.sort()
	triples.sort()
    
	return triples

# Function 8 
# LongestCommonsSubsequence
def longestCommonSubsequence(str1, str2):
    substr1 = subsequences(str1)
    substr2 = subsequences(str2)
    result = {tuple(i):len(i) for i in substr1 if i in substr2}
    result = max(result.items(), key = lambda x: x[1])
    result = list(result[0])
    
    return result

def subsequences(str1):
    subsequence = [[]]
    for item in str1:
        for sub in subsequence:
            subsequence = subsequence +[sub + [item]]
            
    return subsequence

str1= "ZXVVYZW"
str2= "XKYKZPW"

print(longestCommonSubsequence(str1,str2))

# Function 9
# SmallestDifference between pairs considering two different lists
def smallestDifference(arrayOne, arrayTwo):
    
    pairs = {(arrayOne[i],arrayTwo[j]):abs(arrayOne[i] - arrayTwo[j])
             for i in range(len(arrayOne)) for j in range(len(arrayTwo))}
    
    result = min(pairs.items(), key = lambda x:x[1])
    return list(result[0])
	
# Function 10
# Move an element to the end of the list
def moveElementToEnd(array, toMove):
    # Write your code here.
    num_times = 0 # array.count(toMove)
    for i in array:
        if i ==toMove:
            num_times+=1
    
    array = list(filter(lambda x: x!=toMove, array)) + [toMove]*num_times
    return array

def moveElementToEnd(array, toMove):
    # Write your code here.
    num_times = array.count(toMove)
    
    array = list(filter(lambda x: x!=toMove, array)) + [toMove]*num_times
    return array

array = [1,4,5,7,2,2,2,4,4,4]
toMove =4
print(moveElementToEnd(array, toMove))

#Function11
# Monotonic = if its elements are  entirely non-decreasing
# or non-increasing
def isMonotonic(array):
    # Write your code here.
    t = array.copy()
    t.sort()
    if array == t or array == t[::-1]:
	return True
    else:
	return False

# Function12
# Calculate all the permutations of a list
# I can do with itertools but also with below code

elements = [1,2,3]
def permutations(lst):

    if len(lst)==0:
        return lst
    elif len(lst)==1:
        return [lst]

    else:
        result = []
        for i in range(len(lst)):
            m = lst[i]
            rm_list = lst[:i] + lst[i+1:]

            for p in permutations(rm_list):
                result.append([m]+p)

    return result
print(permutations(elements))

# Function 13
# Longest Palindrom Substrings
def ispalindrome(string):
    return string==string[::-1]

def longestPalindromicSubstring(string):

    substringa = {string[i:j]:len(string[i:j]) for i in range(len(string))
              for j in range(i+1,len(string)+1) if ispalindrome(string[i:j])}


    return max(substringa.items(), key = lambda x: x[1])[0]


# Function 14
# fourNumberSum
def fourNumberSum(array, targetSum):
    quadruplets = [[array[i],array[j],array[z],array[m]] 
				   for i in range(len(array))
				   for j in range(i+1,len(array))
				   for z in range(j+1,len(array))
				   for m in range(z+1, len(array))
				   if array[i] + array[j] +array[z] +array[m] ==targetSum]
	
	return quadruplets

# function 15
# levenshteinDistance
def levenshteinDistance(str1,str2):
	edits = [[x for x in range(len(str1)+1)] for y in range(len(str2)+1)]
	for i in range(1, len(str2)+1):
		edits[i][0] = edits[i-1][0]+1
	for i in range(1,len(str2)+1):
		for j in range(1,len(str1)+1):
			if str2[i-1]==str1[j-1]:
				edits[i][j] = edits[i-1][j-1]
			else:
				edits[i][j] = 1+min(edits[i-1][j-1],edits[i-1][j],edits[i][j-1])
	return edits[-1][-1]


# function 16
# maxSubsetSumNoAdjacent
def maxSubsetSumNoAdjacent(array):
    # Write your code here.
    if len(array)==0:
        return 0
    elif len(array)==1:
        return array[0]
    
    maxSums = array[:]
    maxSums[1] = max(array[0], array[1])
    for i in range(2,len(array)):
        maxSums[i] = max(maxSums[i-1], maxSums[i-2] + maxSums[i])
        # print( max(maxSums[i-1], maxSums[i-2] + maxSums[i]))
        # print(maxSums)
        # print(10*'-')
    return maxSums[-1]


# balancedBrackets
# function 17
def find_occurrences(char,str1):
    '''
    '''
    occ = len([i for i in range(len(str1)) if str1.startswith(char,i)])
    return occ


def balancedBrackets(str1):
    '''
    '''
    round_left = '('
    round_right = ')'
    bracket_left = '['
    bracket_right = ']'
    brace_left = '{'
    brace_right = '}'
    
    if find_occurrences(round_left,str1) == find_occurrences(round_right,str1):
        if find_occurrences(bracket_left,str1) == find_occurrences(bracket_right,str1):
            if find_occurrences(brace_left,str1) == find_occurrences(brace_right,str1):
                return True
            else:
                return False
        else:
            return False
    else:
        return False
#### correct one considering the order as well:
def balancedBrackets(string):
    openingBrackets = '([{'
	closingBrackets = ")]}"
	matchingBrackets = {")":"(","]":"[","}":"{"}
	stack = []
	for char in string:
		if char in openingBrackets:
			stack.append(char)
		elif char in closingBrackets:
		     if len(stack)==0:
			return False
		     if stack[-1]==matchingBrackets[char]:
			stack.pop()
		     else:
			return False
	return len(stack)==0
