# load in files
import numpy as np
import sys
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from HMM_helper import *


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

def load_spenser(filename):
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
            elif len(line) <= 6:
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

def load_spenser_sentences(filename):
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
            elif len(line) <= 6:
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

def pre_processing_sentences_reverse_spenser(filename_sha, filename_spen):
    count = 0
    puncs = list('\'!()[]{};:"\\,<>./?@#$%^&*_~') 
    rev_sentences_origin = load_shakespeare_sentences_reverse(filename_sha)
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
    
    
    rev_sentences_spenser = load_spencer_sentences_reverse(filename_spen)
    rev_sentences_spenser = lower_case(rev_sentences_spenser)
    with_puncs_tokens_spenser = pre_tokenize(rev_sentences_spenser)
    no_puncs_tokens_spenser = remove_punc(with_puncs_tokens_spenser)
    for no_puncs_token in no_puncs_tokens_spenser:
        for item in no_puncs_token:
            if item in puncs:
                count += 1
                print("*********************** Warning ***********************")
                print("There is still punctions in pre_processing_sentences-no_puncs_tokens!")
                print("The punction is: ", item)
                print("Is this punction in puncs? ", item in puncs)
                print("*********************** Warning End ***********************")
    
    obs_map = rev_obs_map
    obs = rev_obs
    index = len(obs_map)
    for line in no_puncs_tokens_spenser:
        curr_line = []
        for word in line:
            if word not in obs_map:
                obs_map[word] = index
                index += 1
            curr_line.append(obs_map[word])
        obs.append(curr_line)
        
    return obs, obs_map

# for hmm
def generate_rhyme(model, rhyme_dict, syllable_dict, syllable_end_dict, obs_map_p):
    poem = []
    non_dup = set()
    count = 0
    while count < 7:
        sentence1 = sample_sentence(model, obs_map_p, n_words=10)
        words1 = sentence1.split(" ")
        if (words1[0] in rhyme_dict) and (words1[0] not in non_dup):
            valid1, valid_sentence1 = exact_ten_syllables(words1, syllable_dict, syllable_end_dict)
            while valid1:
                sentence2 = sample_sentence(model, obs_map_p, n_words=10)
                words2 = sentence2.split(" ")
                #print("================ words1[0] = ", words1[0], ", words2[0] = ", words2[0], "=========================")
                if (words2[0] in rhyme_dict[words1[0]]) and (words2[0] not in non_dup):
                    valid2, valid_sentence2 = exact_ten_syllables(words2, syllable_dict, syllable_end_dict)
                    if valid2:
                        poem.append(valid_sentence1)
                        poem.append(valid_sentence2)
                        non_dup.add(words1[0])
                        non_dup.add(words2[0])
                        count += 1
                        break
    
    for i in range(14):
        poem[i] = reverse(poem[i])
    
    poem_output = poem[0] +","+ poem[2] +","+ poem[1] +","+ poem[3] +","+ poem[4] +","+ poem[6] +","+ poem[5] +","+ poem[7] + ","+ \
                  poem[8] +","+ poem[10] +","+ poem[9] +","+ poem[11] +","+ poem[12] +","+ poem[13] +"."
    return poem_output

def reverse(sentence):
    words = sentence.split(" ")
    left = 0
    right = len(words)-1
    
    while left < right:
        temp = words[left]
        words[left] = words[right]
        words[right] = temp
        left += 1
        right -= 1
    
    sentence_reversed = " ".join(words)
    return sentence_reversed

def exact_ten_syllables(words, syllable_dict, syllable_end_dict):
    count = 0
    valid_words = []
    valid = False
    for word in words:
        curr_syllable_max = 0
        curr_syllable_min = 0
        
        if word in syllable_end_dict:
            if count + syllable_end_dict[word] == 10:
                valid_words.append(word)
                valid = True
                break
        
        if len(syllable_dict[word]) == 0:
            break
        elif len(syllable_dict[word]) == 1:
            curr_syllable_len = syllable_dict[word][0]
        else:
            print(len(syllable_dict[word]))
            rnd_idx = random.randint(0, len(syllable_dict[word])-1)
            curr_syllable_len = syllable_dict[word][rnd_idx]
        
        count += curr_syllable_len
        if count < 10:
            valid_words.append(word)
        elif count == 10:
                valid_words.append(word)
                valid = True
                break
        else:
            valid = False
    
    valid_sentence = ""
    if valid:
        for word in valid_words:
            valid_sentence += word + " "
    
    return valid, valid_sentence.strip()




if __name__ == '__main__':
    filename = "./project3/data/shakespeare.txt"
    obs_p, obs_map_p = pre_processing_poems(filename)
    #obs_s, obs_map_s = pre_processing_sentences(filename)
    #print(obs_map_s)
