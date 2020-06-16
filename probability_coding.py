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
    
