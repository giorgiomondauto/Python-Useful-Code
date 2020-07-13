# Question 1: Generate 3 random integers between 100 and 999 which is divisible by 5

import random
def generate_random(start,end,divisible):
    
    numbers = [i for i in range(start,end,1)]
    number_generated = []
    i = 0
    while i < 3:
        new_random = int(random.random() * len(numbers)) 
        if new_random % 5 == 0:
            number_generated.append(numbers[new_random])
            i +=1
        else:
            i += 0
    return number_generated


print(generate_random(100,999,5))

# or easily
random.randrange(100, 999, 5)     

############ Easier Alternative #################
def generate_random(start,end,divisible):
    '''
    '''
    values = [i for i in range(start,end)]
    values = [i for i in values if i%divisible==0]
    random_choice = int(random.random() * len(values))
    value_chosen = values[random_choice]
    
    return value_chosen


print(generate_random(100,999,5))
    
##############################################
# How do you find the number of all distinct permutations of a string? (solution)

# Calculate the Factorial of a Number
def factorial(n):
    count = 1
    for i in range(2,n+1):
        count = count*i
    return count
print(factorial(4))

or 

def factorial(n):
    '''
    '''
    value = n
    i = 1
    while i < n:
        value *= n-i
        i+=1
    return value
print(factorial(5))

##############################################

def numb_perm(str1):
    '''
    len(str1)!/duplicate(char)!
    '''
    # find char with multiple occurrences
    values = {i:str1.count(i) for i in str1 if str1.count(i) >1}
    denominator = 1
    for i in values.values():
        denominator = denominator * factorial(i)
    print(denominator)
    
    result = int(factorial(len(str1))/denominator)
    return result

print(numb_perm('ybghjhbuytb'))
# Input : ybghjhbuytb
# Output : 1663200
# In second example, number of character is 11 and here h and y are repeated 2 times whereas g is repeated 3 times.
# So, number of permutation is 11! / (2!2!3!) = 1663200
