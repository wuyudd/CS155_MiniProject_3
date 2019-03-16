# This file is to split the words_of_poem into lines according to the exactly 10 syllables.
import random
from syllable_dictionary import *
from collections import deque

def split_poem_syllables(poem, syllable_dict, syllable_end_dict):
    poem = poem.split(" ") # get a list of words of this poem
    #print("poem list = ", poem)
    num_sentence = 0 # we need it == 14
    new_poem = [] # final poem, list of lines
    word_th = 0 # current word
    num_lines = 14 # we need 14 lines
    num_syllables = 10

    while(num_sentence < num_lines):
        # new sentence starts
        syllable_count = 0
        syllable_count_list = deque([])
        sentence = deque([])

        # until syllable_count == 10, save the new sentence string to new_poem list.
        while(syllable_count < 10): # haven't reach the end
            curr_word = poem[word_th]
            sentence.append(curr_word)

            curr_word_syllable = get_num_syllables(curr_word, syllable_dict)
            syllable_count += curr_word_syllable
            syllable_count_list.append(curr_word_syllable)

            word_th += 1
            
            # if we reach the end of this sentence (syllable num >= 10)
            if syllable_count >= num_syllables : # reach the end
                if curr_word in syllable_end_dict: # curr_word has an end syllable
                    syllable_count -= syllable_count_list[-1]
                    curr_word_syllable_end = syllable_end_dict[curr_word]
                    if syllable_count + curr_word_syllable_end == num_syllables:
                        break
                    else:
                        discard_word = sentence.popleft()
                        discard_word_count = syllable_count_list.popleft()
                        syllable_count -= discard_word_count
                        continue
                else:
                    if syllable_count + curr_word_syllable == num_syllables:
                        break
                    else:
                        discard_word = sentence.popleft()
                        discard_word_count = syllable_count_list.popleft()
                        syllable_count -= discard_word_count
                        continue

        new_poem.append(' '.join(sentence))
        num_sentence += 1
    
    return new_poem

def get_num_syllables(curr_word, syllable_dict):
    # get the num of syllable of this word in syllable_dict
    curr_word_syllable = 0

    if len(syllable_dict[curr_word]) > 1: # more than one syllables, then random choice
        rnd_idx = random.randint(0, len(syllable_dict[curr_word])-1)
        curr_word_syllable = syllable_dict[curr_word][rnd_idx]
    elif len(syllable_dict[curr_word]) == 1: # only one syllable
        curr_word_syllable = syllable_dict[curr_word][0]
    else:
        curr_word_syllable = 2
        print("*********************** Warning ***********************")
        print("There is an error in the syllable of this word!")
        print("Current word is: ", curr_word, ", Syllable of current word = ", syllable_dict[curr_word])
        print("*********************** Warning ***********************")
        
    return curr_word_syllable

def generate_new_poem(poem, syllable_filename):
    syllable_dict, syllable_end_dict = get_syllable_dict(syllable_filename)
    new_poem = split_poem_syllables(poem, syllable_dict, syllable_end_dict)
    return new_poem

if __name__ == '__main__':
    poem = "when dignity damasked i may and the time nothing in thy grief prove as great give look mine in temperate sweets hath are the lies when one hymn plagues compounds in main dross doth or uphold eye love in happy that transport height so something muse that lamb i call thy being think all a it not term say write not fairer age new own held i believed no from bright thou day lie time know these tattered and checked now deceased now shame all no if tie that made womb a tired that rude was strife fountains sun's fair leaves cry to time let worthy heart men that self suspect prouder maintain cure hand i that do i and and the in come saw love's holy that quality a verse and smother bad the brand old kill captain and thee and my beauteous separation common twice against thou glory a death in fast in new-found write allow gross thereby not memory like perhaps hast the world rich to both sweet strangely power what's to the another's that tripping as lies by trenches or eclipse hides when dear an and doth it thy i indeed too clouds can to not"
    syllable_filename = "./project3/data/Syllable_dictionary.txt"
    new_poem = generate_new_poem(poem, syllable_filename)
    print(new_poem)
