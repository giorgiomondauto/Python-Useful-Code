import json
import glo
import pickle

# to merge dictionaries
def merge_dicts(*dict_args):
    """
    Merge dictionaries
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

# to read dict file
def open_dict(file_name):
    ''' to open dict file '''
    with open(file_name, 'br') as f:
        dictionary = pickle.load(f)
    return dictionary

# from text to json files
def process_text_to_json(file_name, topic_name):
    file_list = []
    with open(file_name + '.txt') as f:
        for line in f:
            line = line.strip().split('-')
            file_list.append({"Title": line[0], "Type": line[1], "Source": line[2], "Link": '-'.join(line[3:])})

    output_name = {topic_name: file_list}
    out_file = open(file_name + '.json', "w") 
    json.dump(output_name, out_file, indent = 4, sort_keys = False) 
    out_file.close()
    return json.dumps(output_name)

# process_text_to_json(file_name = 'miscellaneous', topic_name = 'Miscellaneous')


# merge jsons files
result = []
for f in glob.glob("*.json"):
    with open(f, "rb") as infile:
        result.append(json.load(infile))

with open("merged_file.json", "wb") as outfile:
        json.dump(result, outfile)

# to create a new dictionary starting from a 'x' value
new_dict = dict((v,k) for k,v in enumerate(set(list(\
                                 itertools.chain(*data[columns].apply(lambda x: x.split(' '))))),  max(existing_dictionary.values()) + 1)) # instead of max(  ) + 1 we can type any number


# to remove an empty space from a list
while("" in professioni.Subgroup.iloc[1]) : 
    professioni.Subgroup.iloc[1].remove("")

# to display all the columns of dataframe
pd.set_option('display.max_columns', None)

# convert list of string into a list for a column
import ast
info_data['Info'] = info_data['Info'].apply(lambda x: ast.literal_eval(x))

# to check if a url exists:
import requests
def check_url(url):
    request = requests.get(url)
    if request.status_code == 200:
        print('Web site exists')
        url = url
    else:
        raise Exception('Web site does not exist')
    return url

# to compare urls
def url_compare(new_url,url_list):
    for i in url_list:
        url_base = urlparse(i)
        new_url_test = urlparse(new_url)

        if ((url_base.netloc == new_url_test.netloc) and (url_base.path == new_url_test.path)):
            print('here')
            raise Exception('This Course is already present!!!!')
        else:
            print('Yes. This course can be added')
    return new_url_test

# to read a subset of the csv file
pd.read_csv('data.csv').sample(num_to_load = 100, random_state = 1234)

# to split a column text with multiple delimiters
data['column'].apply(lambda x: re.split('; |, |\*|\n|/',x))

# if startswith multiple values - text in a column
prefixes = ['di','dal','nan','del','sug','nan','dei']
data['column'].apply(lambda x: [i for i in x if not i.startswith(tuple(prefixes))])

# create dictionary from two columns
pd.Series(source_hospital['ZIP Code'].values,index=source_hospital['Facility Name']).to_dict()

# to calculate the distance between two text
mport numpy as np
def levenshtein_ratio_and_distance(s, t, ratio_calc = False):
    """ levenshtein_ratio_and_distance:
        Calculates levenshtein distance between two strings.
        If ratio_calc = True, the function computes the
        levenshtein distance ratio of similarity between two strings
        For all i and j, distance[i,j] will contain the Levenshtein
        distance between the first i characters of s and the
        first j characters of t
    """
    # Initialize matrix of zeros
    rows = len(s)+1
    cols = len(t)+1
    distance = np.zeros((rows,cols),dtype = int)

    # Populate matrix of zeros with the indeces of each character of both strings
    for i in range(1, rows):
        for k in range(1,cols):
            distance[i][0] = i
            distance[0][k] = k

    # Iterate over the matrix to compute the cost of deletions,insertions and/or substitutions    
    for col in range(1, cols):
        for row in range(1, rows):
            if s[row-1] == t[col-1]:
                cost = 0 # If the characters are the same in the two strings in a given position [i,j] then the cost is 0
            else:
                # In order to align the results with those of the Python Levenshtein package, if we choose to calculate the ratio
                # the cost of a substitution is 2. If we calculate just distance, then the cost of a substitution is 1.
                if ratio_calc == True:
                    cost = 2
                else:
                    cost = 1
            distance[row][col] = min(distance[row-1][col] + 1,      # Cost of deletions
                                 distance[row][col-1] + 1,          # Cost of insertions
                                 distance[row-1][col-1] + cost)     # Cost of substitutions
    if ratio_calc == True:
        # Computation of the Levenshtein Distance Ratio
        Ratio = ((len(s)+len(t)) - distance[row][col]) / (len(s)+len(t))
        return Ratio
    else:
        # print(distance) # Uncomment if you want to see the matrix showing how the algorithm computes the cost of deletions,
        # insertions and/or substitutions
        # This is the minimum number of edits needed to convert string a to string b
        return "The strings are {} edits away".format(distance[row][col])
 
Str1 = "Apple Inc.aaaaa"
Str2 = "apple Inc"
Distance = levenshtein_ratio_and_distance(Str1.lower(),Str2.lower())
print(Distance)
Ratio = levenshtein_ratio_and_distance(Str1.lower(),Str2.lower(),ratio_calc = True)
print(Ratio)

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

# rename columns in one go with a range of int
new.rename(columns=lambda x: 'diag'+str(x), inplace = True)

# to read and concatenate multiple files at once
import glob
df = pd.concat(map(functools.partial(pd.read_csv, sep='|', index_col=0), glob.glob(os.path.join(INPATH, "parsed*") ) ))


# map columns from dictionary
edi['PatZip5'].apply(lambda x: county_dict.get(x,'Unknown')).value_counts()


# to map data from column without creating a dictionary
 for col in ['Patient']:
            adt = pd.merge(adt, ccodes, left_on = col+'ZipCode', 
                           right_on = 'ZIP', 
                           how = 'left')\
                           .rename(columns={'COUNTYNAME':col+'County'})\
                           .drop(['ZIP'], 1)
                           
# to check if all chars of a substring are in a string
def f(s, Substring):
     return all(s.count(i)>=Substring.count(i) for i in Substring)                            
                           

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


#### all the combinations of a set without combinations from itertools ####
def my_function(arr):
    '''
    '''
    result = [[]]
    for item in arr:
        for subset in result:
            result = result + [list(subset) + [item]]
            
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
