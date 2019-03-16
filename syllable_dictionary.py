from collections import defaultdict
from nltk.corpus import cmudict
import nltk
nltk.download('cmudict')
import pyphen

def read_dict(filename):
    words = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            words.append(line)
    return words

def dictionary(words):
    syllable_dict = defaultdict(list)
    syllable_end_dict = {}
    for word in words:
        temp = word.split( )
        key = temp[0]
        for i in range(1, len(temp)):
            if temp[i].isdigit():
                syllable_dict[key].append(int(temp[i]))
            else:
                syllable_end_dict[key]=int(temp[i][1:])
    return syllable_dict, syllable_end_dict


def get_syllable_dict(filename):
    words = read_dict(filename)
    syllable_dict, syllable_end_dict = dictionary(words)
    return syllable_dict, syllable_end_dict

def add_spenser_syllable_dict(words, syllable_dict):
    for word in words:
        if(word not in syllable_dict):
            #s = []
            if word.lower() in cmudict.dict().keys():
                syllable_dict[word] = look_thru_cmu(word)
                #print("in cmudict")
            else:
                syllable_dict[word].append(look_thru_pyphen(word))
                #print("in pyphen")
    return syllable_dict

def look_thru_cmu(word):
    d = cmudict.dict()
    return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]] 

def look_thru_pyphen(word):
    dic = pyphen.Pyphen(lang='en')
    parsedWord = dic.inserted(word)
    return parsedWord.count('-') + 1

if __name__ == '__main__':
	filename = "./project3/data/Syllable_dictionary.txt"
	syllable_dict, syllable_end_dict = get_syllable_dict(filename)
	
