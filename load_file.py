# load in files

def load_shakespeare(filename):
    '''
    peoms
    '''
    peoms = []
    count = 0
    with open(filename, 'r') as f:
        peom = ""
        for line in f.readlines():
            if line[-1] != "\n": # the last peom
                peoms.append(peom)
            line = line.strip()
            if line.isdigit(): # digit for different peoms
                continue
            elif len(line) == 0: # 2 empty line, this peom ends
                count += 1
                if count == 2:
                    peoms.append(peom.strip())
                    peom = ""
                    count = 0
            else:
                line += " "
                peom += line
    return peoms

def load_shakespeare_sentences(filename):
    '''
    sentences
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