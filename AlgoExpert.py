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
# This is the class of the input root. Do not edit it.
# This is the class of the input root. Do not edit it.
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
	
    
