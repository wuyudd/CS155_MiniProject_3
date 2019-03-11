import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Lambda
from keras.layers.recurrent import LSTM
from keras.preprocessing.sequence import pad_sequences
from pre_processing import *

def load_text_as_string(filepath):
    poems_ori = load_shakespeare_sentences(filepath)
    poems = lower_case(poems_ori)
    text = ''
    #print(poems)
    for line in poems:
            text += line + '\n'
    return text

def run_rnn(X, y, temperature, batch_size, epochs):
    model = Sequential()
    model.add(LSTM(200, return_sequences=True, input_shape=(len(X[0]), len(X[0][0]))))
    model.add(Dropout(0.2))
    model.add(LSTM(200, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(len(X[0][0])))
    model.add(Activation('softmax'))
    model.add(Lambda(lambda x: x / temperature))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X, y, batch_size = batch_size, epochs = epochs )
    
    error_in = model.evaluate(X, y, batch_size = batch_size)
    return model, error_in

def generate_train_data(text, step):
    char_list = list(set(text))
    char_dict = {}
    index = 0
    for c in char_list:
        char_dict[c] = index
        index += 1
    char_num = len(char_dict)
    
    seq_length = 40
    X_train_string = []
    Y_train_string = []
    for i in range(0, len(text)-seq_length, step):
        sentence = text[i:i+seq_length]
        next_char = text[i+seq_length]
        X_train_string.append(sentence)
        Y_train_string.append(next_char)    
    
    X_train_encode = []
    Y_train_encode = []
    for sentence in X_train_string:
        new_sentence = []
        for c in sentence:
            new_sentence.append(char_dict[c])
        X_train_encode.append(new_sentence)
    for next_char in Y_train_string:
        Y_train_encode.append(char_dict[next_char])
    
    X_train = np.array([to_categorical(seq, num_classes = char_num) for seq in X_train_encode])
    Y_train = to_categorical(Y_train_encode, num_classes = char_num)
    
    return X_train, Y_train, char_dict, char_num

def generate_sequence(model, char_dict, seed_text, num_chars, max_prob=False):
	char_dict_inverse = {}
	for key, val in char_dict.items():
		char_dict_inverse[val] = key

	# initlization
	new_text = seed_text
	for char in range(num_chars):
		sequence = [char_dict[char] for char in new_text]
		sequence = pad_sequences([sequence], maxlen=40)
		encoded_seq = to_categorical(sequence, num_classes=len(char_dict))

		if max_prob:
			pred_class = model.predict_classes(encoded_seq)
			# print("************************* pred max! *************************")
			# print("max_prob pred_class = ", pred_class)
			# print("************************* pred max! end *************************")
		else:
			pred_classes = model.predict(encoded_seq)[0]
			# print("************************* pred no max! *************************")
			# print("no max_prob pred_classes = ", pred_classes)
			# print("************************* pred no max! end *************************")
			
			normalized_pred_classes = []
			sum_pred = sum(pred_classes)

			for predict in pred_classes:
				normalized_pred_classes.append(predict / sum_pred)

			pred_class = np.random.choice(range(len(normalized_pred_classes)), p=normalized_pred_classes)
			# print("************************* pred no max! *************************")
			# print("no max_prob pred_class = ", pred_class)
			# print("************************* pred no max! end *************************")

		new_char = char_dict_inverse[pred_class]
		new_text += new_char

	return new_text


def rnn(filepath):
	#filepath = './project3/data/shakespeare.txt'
	text = load_text_as_string(filepath)
	step = 5
	temperature = 1.5
	batch_size = 32
	epochs = 20
	seed_text = "shall i compare thee to a summer's day?\n"
	num_chars = 3000

	X_train, Y_train, char_dict, char_num = generate_train_data(text, step)
	print(char_dict)

	model, error_in = run_rnn(X_train, Y_train, temperature, batch_size, epochs)
	new_text = generate_sequence(model, char_dict, seed_text, num_chars)

	return error_in, new_text


if __name__ == '__main__':
	filepath = './project3/data/shakespeare.txt'
	error_in, new_text = rnn(filepath)
	print("************************* Final *************************")
	print("Error_In = ", error_in)
	print("new text: ", new_text)
	print("************************* Final End *************************")
