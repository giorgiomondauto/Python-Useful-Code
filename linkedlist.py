####### LINKED LIST ########
class linkedListNode:
    def __init__(self, value, nextNode=None):
        self.value = value
        self.nextNode = nextNode

def insertNode(head, valuetoInsert):
    currentNode = head
    while currentNode is not None:
        if currentNode.nextNode is None:
            currentNode.nextNode = linkedListNode(valuetoInsert)
            return head
        currentNode = currentNode.nextNode

# Delete node function
def deleteNode(head, valueToDelete):
    currentNode = head
    previousNode = None
    while currentNode is not None:
        if currentNode.value == valueToDelete:
            if previousNode is None:
                newHead = currentNode.nextNode
                currentNode.nextNode = None
                return newHead
            previousNode.nextNode = currentNode.nextNode
            return head
        previousNode = currentNode
        currentNode = currentNode.nextNode
    return head  # Value to delete was not found.

# "3" -> "7" -> "10"

node1 = linkedListNode("3") # "3"
node2 = linkedListNode("7") # "7"
node3 = linkedListNode("10") # "10"

node1.nextNode = node2 # node1 -> node2 , "3" -> "7"
node2.nextNode = node3 # node2 -> node3 , "7" -> "10"

# node1 -> node2 -> node3

head = node1
print ("*********************************")
print ("Traversing the regular linkedList")
print ("*********************************")
# Regular Traversal
currentNode = head
while currentNode is not None:
    print (currentNode.value),
    currentNode = currentNode.nextNode
print ('')
print( "*********************************")

print ("deleting the node '7'")
newHead = deleteNode(head, "10")

print ("*********************************")
print ("traversing the new linkedList with the node 7 removed")
print ("*********************************")

currentNode = newHead
while currentNode is not None:
    print(currentNode.value),
    currentNode = currentNode.nextNode

print ('')
print ("*********************************")
print ("Inserting the node '99'")
newHead = insertNode(newHead, "99")

print ("*********************************")
print ("traversing the new linkedList with the node 99 added")
print ("*********************************")

currentNode = newHead
while currentNode is not None:
    print (currentNode.value),
    currentNode = currentNode.nextNode
    
 ####### FIND MIDDLE OF A LINKED LIST ########
class Node: 
    def __init__(self, value): 
        self.data = value 
        self.next = None
      
class LinkedList: 
  
    def __init__(self): 
        self.head = None
  
    # create Node and and make linked list 
    def push(self, new_data): 
        new_node = Node(new_data) 
        new_node.next = self.head 
        self.head = new_node 
          
    def printMiddle(self): 
        temp = self.head  
        count = 0
          
        while self.head: 
  
            # only update when count is odd 
            if (count & 1):  
                temp = temp.next
            self.head = self.head.next
  
            # increment count in each iteration  
            count += 1 
          
        print(temp.data)  
        
list1 = LinkedList() 
list1.push(5) 
list1.push(4) 
list1.push(2) 
list1.push(3) 
list1.push(1) 
list1.printMiddle()
