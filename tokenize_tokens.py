# tokenize
# two ways: one keeps the punctions, one doesn't.

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

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