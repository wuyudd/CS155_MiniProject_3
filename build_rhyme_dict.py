from pre_processing import *
import collections

def read_in_poems(filename):
	poems = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if line.isdigit(): # new poem starts
				new_poem = []
				count = 0
			elif len(line) == 0:
				count += 1
				if count == 2: # this poem ends
					poems.append(new_poem)
			else:
				new_poem.append(line)
	return poems

def read_in_poems_sepnser(filename):
	poems = []
	with open(filename, 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if len(line) < 10 and len(line) != 0: # new poem starts
				new_poem = []
				count = 0
			elif len(line) == 0:
				count += 1
				if count == 2: # this poem ends
					poems.append(new_poem)
				new_poem = []
				count = 0
			else:
				new_poem.append(line)
	return poems

def get_last_word(poems):
	all_poems_last_words = []
	for poem in poems:
		if len(poem) != 14:
			continue
		poem_last_words = []
		for i in range(len(poem)):
			last_word = poem[i].split(" ")[-1]
			if last_word.isalpha():
				poem_last_words.append(last_word)
			else:
				poem_last_words.append(last_word[:-1])
		all_poems_last_words.append(poem_last_words)

	return all_poems_last_words


def build_rhyme_dict(all_poems_last_words):
	rhyme_dict = collections.defaultdict(list)
	for poem_last_words in all_poems_last_words:
		if len(poem_last_words) != 14:
			print("Error: This poem doesn't contain 14 lines!")
		else:
			#print(poem_last_words)
			for i in range(len(poem_last_words)):
				if i == 0 or i == 1 or i == 4 or i == 5 or i == 8 or i == 9:
					rhyme_dict[poem_last_words[i]].append(poem_last_words[i+2])
					rhyme_dict[poem_last_words[i+2]].append(poem_last_words[i])
				elif i == 12:
					rhyme_dict[poem_last_words[i]].append(poem_last_words[i+1])
					rhyme_dict[poem_last_words[i+1]].append(poem_last_words[i])

	return rhyme_dict

def get_rhyme_dict(filename):
	poems = read_in_poems(filename)
	all_poems_last_words = get_last_word(poems)
	rhyme_dict = build_rhyme_dict(all_poems_last_words)

	return rhyme_dict

def get_rhyme_dict_spenser(filename):
	poems = read_in_poems_sepnser(filename)
	all_poems_last_words = get_last_word(poems)
	rhyme_dict = build_rhyme_dict(all_poems_last_words)

	return rhyme_dict

def combine_sha_spen_rhyme_dict(sha_rhyme_dict, spen_rhyme_dict):
	comebine_rhyme_dict = {}
	for key_spen, value_spen in spen_rhyme_dict.items():
		if key_spen in sha_rhyme_dict.keys():
			new_rhyme_words = list(set(value_spen + sha_rhyme_dict[key_spen]))
			comebine_rhyme_dict[key_spen] = new_rhyme_words
		else:
			comebine_rhyme_dict[key_spen] = value_spen

	for key_sha, value_sha in sha_rhyme_dict.items():
		if key_sha not in spen_rhyme_dict.keys():
			comebine_rhyme_dict[key_sha] = value_sha

	print("!!! IN build_rhyme_dict.py function: combine_sha_spen_rhyme_dict")
	print("1==length of sha_rhyme_dict = ", len(sha_rhyme_dict))
	print("2==length of spen_rhyme_dict = ", len(spen_rhyme_dict))
	print("3==length of comebine_rhyme_dict = ", len(comebine_rhyme_dict))
	print("!!! OUT build_rhyme_dict.py function: combine_sha_spen_rhyme_dict")
	return comebine_rhyme_dict

if __name__ == '__main__':
	filename = "./project3/data/shakespeare.txt"
	poems = read_in_poems_sepnser(filename)
	#print(poems[0])
	all_poems_last_words = get_last_word(poems)
	#print(all_poems_last_words[0])
	rhyme_dict = build_rhyme_dict(all_poems_last_words)
	print(rhyme_dict)
	
