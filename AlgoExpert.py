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

