# naive HMM model for shakespeare
from pre_processing import *
import os
import numpy as np
from IPython.display import HTML

from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)

filename = "./project3/data/shakespeare.txt"
obs, obs_map = pre_processing_res(filename)
hmm8 = unsupervised_HMM(obs, 10, 100)
visualize_sparsities(hmm8, O_max_cols=50)
print('Sample Sentence:\n====================')
print(sample_sentence(hmm8, obs_map, n_words=25))


