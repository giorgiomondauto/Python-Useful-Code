# permutations - combinations between two lists
a = ["foo", "melon"]
b = ['ciao', 'yes','si']
c = list(itertools.product(a, b))

# to get all the divisors of a number
def printDivisors(n) : 
    i = 1
    result = []
    while i <= n : 
        if (n % i==0) : 
            result.append(i) 
        i = i + 1
    return result
printDivisors(20)

# bubble_sort
def bubble_sort(values):
    for i in range(len(values)):
        for j in range(0, len(values)-i-1):
            if values[j] > values[j+1]:
                values[j],values[j+1] = values[j+1], values[j]
                
    return values
print(bubble_sort([64, 34, 25, 12, 22, 11, 90]))

###### consecutive values
def isconsecutive(values):
    n = len(values)+ (min(values)-1)
    sum_prior = sum([i for i in range(min(values))])
    expected_sum = ((n*(n+1))/2)  - sum_prior
    print('expected_sum',expected_sum)
    actual_sum = sum(values)
    print('actual_sum',actual_sum)
    return expected_sum == actual_sum

values1 = [1,2,3,4,5]
value2 = [4,5,6]
print(isconsecutive(value2))

############
# How to find all occurrences of a substring?

str1 = 'hello I am hulk'

print([i for i in range(len(str1)) if str1.startswith('l',i)])


# how to find all occurrences of an item in a list
 print([i for i, x in enumerate(indeces) if x == "26489"])

# Largest subarray with consecutive integers (when integers are consecutives).

def largest_sub(values):
    '''
    Largest subarray with consecutive integers (when integers are consecutives)
    '''
    subarrays = []
    for i in range(len(values) + 1):
        subarrays.append(list(itertools.combinations(values,i)))
    
    result = {i:len(i) for x in subarrays for i in x  if isconsecutive(list(i)) == True}
    
    return max(result.items(), key = lambda x: x[1])

# Subarray - Subsequence - Subset
# Consider an array:
#  {1,2,3,4}
# Subarray: contiguous sequence in an array i.e.

# {1,2},{1,2,3}
# Subsequence: Need not to be contiguous, but maintains order i.e.

# {1,2,4}
# Subset: Same as subsequence except it has empty set i.e.

#  {1,3},{}
# Given an array/sequence of size n, possible

# Subarray = n*(n+1)/2
# Subseqeunce = (2^n) -1 (non-empty subsequences)
# Subset = 2^n

# print(largest_sub([1, 9, 3, 10, 4, 20, 2]))

    
def ispalindrome(word):
    '''
    '''
    return word == word[::-1]
print(ispalindrome('redrum murder'))

def is_palindrome(s):
    if len(s) < 1:
        return True
    else:
        if s[0] == s[-1]:
            return is_palindrome(s[1:-1])
        else:
            return False

        
        
##### all pairs ##### 
values = [9,11,21,23]

result = []
for i in range(len(values)):
    for j in range(i+1, len(values)):
        result.append([values[i],values[j]])
         
print(result)


#### all permutations of a list of words
words = ["foo","bar"]
results = []
for i in range(len(words)):
    for j in range(len(words)):
        if i!=j:
            results.append([''.join([words[i],words[j]])])
        
print(results)


##### all triples #####
L = [-1, 0, 1, 2, -1, -4]

values = []
for i in range(len(L)):
    for j in range(i+1,len(L)):
        for z in range(j+1,len(L)):
            
             values.append([L[i],L[j], L[z]])       
print(values)
######### or
print([(L[i],L[j],L[z]) for i in range(len(L)) for j in range(i+1,len(L)) for z in range(j+1,len(L))])


#### all the subsets of a set without combinations from itertools ####
def my_function(arr):
    '''
    2^n where n is the number of values/len(array)
    '''
    result = [[]]
    for item in arr:
        for subset in result:
            result = result + [subset + [item]]
            
    return result

print(list(my_function([4, 5, 6])))

#### find all the subarrays of an array
def sub_lists(list1): 
  
    # store all the sublists  
    sublist = [[]] 
      
    # first loop  
    for i in range(len(list1) + 1): 
          
        # second loop  
        for j in range(i + 1, len(list1) + 1): 
              
            # slice the subarray  
            sub = list1[i:j] 
            sublist.append(sub) 
            
    return sublist
              
print(sub_lists(values))


#### remove values from a list
nums = [3,2,2,3]

print(list(filter(lambda x: x != 3, nums))) # remove all occurrences
# first occurence
nums.remove(3)
print(nums)
# remove by index
nums.pop(3) # index 3
print(nums)

##### Merge Sorted Array (Python) #####
# Input:
# nums1 = [1,2,3,0,0,0], m = 3
# nums2 = [2,5,6],       n = 3

# Output: [1,2,2,3,5,6]

nums1 = [1,2,3,0,0,0]
nums2 = [2,5,6]

def merge(nums1,nums2, m, n):
    
    del nums1[m:]
    nums1 += nums2[0:n]
    nums1.sort()
    return nums1

print(merge(nums1,nums2,3,3))
