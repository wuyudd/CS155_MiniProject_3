from collections import defaultdict

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

if __name__ == '__main__':
	filename = "./project3/data/Syllable_dictionary.txt"
	syllable_dict, syllable_end_dict = get_syllable_dict(filename)
	
