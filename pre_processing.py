# load in files
import numpy as np
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

def load_shakespeare(filename):
    '''
    input: filename
    output: list of strings, each string is a complete poem.
    '''
    poems = []
    count = 0
    with open(filename, 'r') as f:
        poem = ""
        for line in f.readlines():
            if line[-1] != "\n": # the last poem
                poems.append(poem)
            line = line.strip()
            if line.isdigit(): # digit for different poems
                continue
            elif len(line) == 0: # 2 empty line, this poem ends
                count += 1
                if count == 2:
                    poems.append(poem.strip())
                    poem = ""
                    count = 0
            else:
                line += " "
                poem += line
    return poems

def load_shakespeare_sentences(filename):
    '''
    input: filename
    output: list of strings, each string is a sentence.
    '''
    sentences = []
    count = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) == 0: # empty line
                continue
            elif line.isdigit(): # peom number
                #print("# of peom = ", int(line)-1, ", count = ", count)
                count = 0
                continue
            else: # sentence
                sentences.append(line)
                count += 1
    return sentences 

def lower_case(list_of_strings):
    new_list = []
    for string in list_of_strings:
        new_list.append(string.lower())
    return new_list

# tokenize
# two ways: one keeps the punctions, one doesn't.
def pre_tokenize(poems):
    result = []
    puncs = '\'!()[]{};:"\\,<>./?@#$%^&*_~'
    for poem in poems:
        tokens_nltk = word_tokenize(poem)
        has_s = False    
        #print("tokens_nltk = ", tokens_nltk)
        
        for i in range(len(tokens_nltk)):
            if tokens_nltk[i] == "'s":
                tokens_nltk[i-1] += "'s"
        if "'s" in tokens_nltk:
            tokens_nltk = list(filter(lambda a: a != "'s", tokens_nltk))
        result.append(tokens_nltk)
    return result

def remove_punc(result):
    puncs = '\'!()[]{};:"\\,<>./?@#$%^&*_~'
    for peom_tokens in result:
        for token in peom_tokens:
            if token in puncs:
                peom_tokens.remove(token)
    return result

# encode word
def encode(data):
    '''
    encode word for input for HMM
    '''
    index = 0
    obs = []
    obs_map = {}
    for line in data:
        curr_line = []
        for word in line:
            if word not in obs_map:
                obs_map[word] = index
                index += 1
            curr_line.append(obs_map[word])
        obs.append(curr_line)
    return obs, obs_map

def pre_processing_res(filename):
    # read in poems
    poems_origin = load_shakespeare(filename)
    poems = lower_case(poems_origin)
    # tokenize without puncs
    with_puncs_tokens = pre_tokenize(poems)
    no_puncs_tokens = remove_punc(with_puncs_tokens)

    obs, obs_map = encode(no_puncs_tokens)
    # print("********************* obs *********************")
    # print(obs[0])
    # print("********************* obs_map *********************")
    # print(obs_map)
    return obs, obs_map


if __name__ == '__main__':
    filename = "./project3/data/shakespeare.txt"
    obs, obs_map = pre_processing_res()
    print(obs_map)
