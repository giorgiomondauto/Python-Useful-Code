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
