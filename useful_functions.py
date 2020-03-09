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
