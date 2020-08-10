# Linked Lists
# page 176/540

class linkedlist:
    '''
    '''
    def __init__(self,value,next_node = None):
        '''
        '''
        self.value = value
        self.next_node = next_node
        
        
a = linkedlist(value = 10)
b = linkedlist(value = 5)
c = linkedlist(value = 15)

a.next_node = b
b.next_node = c

head = a
current_node = head
while current_node is not None:
    print(current_node.value)
    current_node = current_node.next_node
        
print(10*'#')
# SEARCHING NODE
# The linked list search operation requires O(n) in the worst case, which occurs when
# the target item is not in the list.
def searching_node(head,target):
    current_node = head
    while current_node is not None and current_node.value != target:
        current_node = current_node.next_node
    return current_node

print(searching_node(a,5))
print(10*'#')

# ADDING A NEW NODE AS HEAD
d = linkedlist(value = 20)
d.next_node = head # a
head = d

current_node = head
while current_node is not None:
    print(current_node.value)
    current_node = current_node.next_node
    
print(10*'#')
# DELETING A NODE
predNode = None
current_node = head
target = 5
while current_node is not None and current_node.value != target:
    predNode = current_node
    current_node = current_node.next_node

if current_node is not None:
    if current_node is head:
        head = current_node.next_node
    else:
        predNode.next_node = current_node.next_node

current_node = head
while current_node is not None:
    print(current_node.value)
    current_node = current_node.next_node
