# load in files
import numpy as np
import sys
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize


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


def load_shakespeare_sentences_reverse(filename):
    '''
    input: filename
    output: list of strings, each string is a reversed sentence. (no end puncs)
    '''
    rev_sentences = []
    #count = 0
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            new_line = ""
            if len(line) == 0: # empty line
                continue
            elif line.isdigit(): # peom number
                #print("# of peom = ", int(line)-1, ", count = ", count)
                #count = 0
                continue
            else: # sentence
                if line[-1].isalpha():
                    words = line.split(" ")
                    rev_words = [words[i] for i in range(len(words)-1, -1, -1)] # reverse
                    new_line = " ".join(rev_words)
                else:
                    words = line[:-1].split(" ")
                    rev_words = [words[i] for i in range(len(words)-1, -1, -1)]
                    new_line = " ".join(rev_words)
            rev_sentences.append(new_line)
                #count += 1
    return rev_sentences 

def lower_case(list_of_strings):
    new_list = []
    for string in list_of_strings:
        new_list.append(string.lower())
    return new_list

def pre_tokenize_full(poems):
    #return list of string
    # result = []
    #puncs = '\'!()[]{};:"\\,<>./?@#$%^&*_~'
    # for poem in poems:
    tokens_nltk = word_tokenize(poems)
    #tokens_nltk = wordpunct_tokenize(poem)
    has_s = False    
    #print("tokens_nltk = ", tokens_nltk)
    
    for i in range(len(tokens_nltk)):
        if tokens_nltk[i] == "'s":
            tokens_nltk[i-1] += "'s"
    if "'s" in tokens_nltk:
        tokens_nltk = list(filter(lambda a: a != "'s", tokens_nltk))
    # result.append(tokens_nltk)
    return tokens_nltk

# tokenize
# two ways: one keeps the punctions, one doesn't.
def pre_tokenize(poems):
    result = []
    #puncs = '\'!()[]{};:"\\,<>./?@#$%^&*_~'
    for poem in poems:
        tokens_nltk = word_tokenize(poem)
        #tokens_nltk = wordpunct_tokenize(poem)
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
    puncs = list('\'!()[]{};:"\\,<>./?@#$%^&*_~')
    new_result = []
    for peom_tokens in result:
        tmp = []
        for token in peom_tokens:
            if token not in puncs:
                tmp.append(token)
        new_result.append(tmp)
    return new_result

# encode word
def encode(data):
    '''
    encode word for input for HMM
    '''
    puncs = list('\'!()[]{};:"\\,<>./?@#$%^&*_~')
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

    for key, value in obs_map.items():
        if key in puncs:
            print("*********************** Warning ***********************")
            print("There is still punctions in the obs_map!!!")
            print("The punction is: ", key)
            print("Is this punction in puncs? ", key in puncs)
            print("*********************** Warning End ***********************")
    return obs, obs_map

def pre_processing_poems(filename):
    # read in poems
    count = 0
    puncs = list('\'!()[]{};:"\\,<>./?@#$%^&*_~')
    poems_origin = load_shakespeare(filename)
    poems = lower_case(poems_origin)
    # tokenize without puncs
    with_puncs_tokens = pre_tokenize(poems)
    #print("with_puncs_tokens:")
    #print(with_puncs_tokens)
    no_puncs_tokens = remove_punc(with_puncs_tokens)
    for no_puncs_token in no_puncs_tokens:
        for item in no_puncs_token:
            if item in puncs:
                count += 1
                print("*********************** Warning ***********************")
                print("There is still punctions in pre_processing_poems-no_puncs_tokens!")
                print("The punction is: ", item)
                print("Is this punction in puncs? ", item in puncs)
                print("*********************** Warning End ***********************")
    print("count = ", count)
    obs, obs_map = encode(no_puncs_tokens)
    # print("********************* obs *********************")
    # print(obs[0])
    # print("********************* obs_map *********************")
    # print(obs_map)
    return obs, obs_map

def pre_processing_sentences(filename):
    count = 0
    puncs = list('\'!()[]{};:"\\,<>./?@#$%^&*_~') 
    sentences_origin = load_shakespeare_sentences(filename)
    sentences = lower_case(sentences_origin)
    with_puncs_tokens = pre_tokenize(sentences)
    no_puncs_tokens = remove_punc(with_puncs_tokens)
    for no_puncs_token in no_puncs_tokens:
        for item in no_puncs_token:
            if item in puncs:
                count += 1
                print("*********************** Warning ***********************")
                print("There is still punctions in pre_processing_sentences-no_puncs_tokens!")
                print("The punction is: ", item)
                print("Is this punction in puncs? ", item in puncs)
                print("*********************** Warning End ***********************")
    obs, obs_map = encode(no_puncs_tokens)
    print("count of puncs = ", count)
    return obs, obs_map

def pre_processing_sentences_reverse(filename):
    count = 0
    puncs = list('\'!()[]{};:"\\,<>./?@#$%^&*_~') 
    rev_sentences_origin = load_shakespeare_sentences_reverse(filename)
    rev_sentences = lower_case(rev_sentences_origin)
    with_puncs_tokens = pre_tokenize(rev_sentences)
    no_puncs_tokens = remove_punc(with_puncs_tokens)
    for no_puncs_token in no_puncs_tokens:
        for item in no_puncs_token:
            if item in puncs:
                count += 1
                print("*********************** Warning ***********************")
                print("There is still punctions in pre_processing_sentences-no_puncs_tokens!")
                print("The punction is: ", item)
                print("Is this punction in puncs? ", item in puncs)
                print("*********************** Warning End ***********************")
    rev_obs, rev_obs_map = encode(no_puncs_tokens)
    print("count of puncs = ", count)
    return rev_obs, rev_obs_map

if __name__ == '__main__':
    filename = "./project3/data/shakespeare.txt"
    obs_p, obs_map_p = pre_processing_poems(filename)
    #obs_s, obs_map_s = pre_processing_sentences(filename)
    #print(obs_map_s)
