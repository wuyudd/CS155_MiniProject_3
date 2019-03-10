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