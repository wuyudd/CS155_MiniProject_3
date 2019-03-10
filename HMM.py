########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.
        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.
        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        #start 
        #compute the sequence
        for word in range(0,M):
            if(word == 0):
                for i in range(self.L):
                    probs[word+1][i] = self.A_start[i]*self.O[i][x[word]]
            else:
                for cur_type in range(self.L):
                    max_value=0
                    max_indx = 0
                    for prev_type in range(self.L): #probs[]starts from 1, probs[word][] is the stored value of prev layer
                        cur_value = probs[word][prev_type]*self.A[prev_type][cur_type]*self.O[cur_type][x[word]]
                        if(cur_value>= max_value):
                            max_value = cur_value
                            max_indx = prev_type
                    probs[word+1][cur_type] = max_value
                    seqs[word+1][cur_type] = max_indx
                if(word == M-1):
                    max_v = max(probs[M])
                    order = str(probs[M].index(max_v))
        
        for word in range(M, 0, -1):
            order += str(seqs[word][int(order[len(order)-1])])
        #reverse the sequence
        max_seq = order[::-1]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bi)
        ###
        ###
        ###

        
        #computing forwarding
        for word in range(0,M):
            for cur_type in range(self.L):
                p = 0
                for prev_type in range(self.L):
                    if(word == 0):
                        alphas[word+1][prev_type] = self.A_start[prev_type]*self.O[prev_type][x[word]]
                    else:
                        p = p + alphas[word][prev_type]*self.A[prev_type][cur_type]*self.O[cur_type][x[word]]

                if(word != 0):
                    alphas[word+1][cur_type] = p
            #normalize
            if(normalize):
                norm_sum = sum(alphas[word+1])
                for norm_type in range(self.L):
                    alphas[word+1][norm_type] = alphas[word+1][norm_type]/norm_sum


        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bii)
        ###
        ###
        ###

        #start
        
        for word in range(0, -M-1, -1):
            for cur_type in range(self.L):
                p=0
                for next_type in range(self.L):
                    if(word == 0):
                        betas[word-1][next_type] = 1
                    else:
                        if (word !=-M):
                            p = p + betas[word][next_type] * self.A[cur_type][next_type] * self.O[next_type][x[word]]
                        else:
                            p = p + betas[word][next_type] * self.A_start[next_type] * self.O[next_type][x[word]]
                if(word != 0):
                    betas[word-1][cur_type] = p
            if(normalize):
                norm_sum = sum(betas[word-1])
                for norm_type in range(self.L):
                    betas[word-1][norm_type] /= norm_sum


        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        
        for cur_type in range(self.L):
            for next_type in range(self.L):
                den_count=0
                num_count=0
                for y in Y:
                    for i in range(len(y)-1):
                        if(y[i]==cur_type and y[i+1]==next_type):
                            num_count += 1
                        if(y[i]==cur_type):
                            den_count += 1
                self.A[cur_type][next_type] = num_count/den_count



        # Calculate each element of O using the M-step formulas.

        for w in range(self.D):
            for z in range(self.L):
                num_count=0
                den_count=0
                for i in range(len(X)):
                    for j in range(len(X[i])):
                        if(X[i][j]==w and Y[i][j]==z):
                            num_count+=1
                        if(Y[i][j]==z):
                            den_count+=1
                self.O[z][w] = num_count/den_count


        pass


    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

            Arguments:
                X:          A dataset consisting of input sequences in the form
                            of lists of length M, consisting of integers ranging
                            from 0 to D - 1. In other words, a list of lists.

                N_iters:    The number of iterations to train on.
        '''
        for iterations in range(N_iters):
            #copy A and O
            #print(iterations)
            A_num = np.zeros((self.L, self.L))
            A_den = np.zeros((self.L,1))
            O_num = np.zeros((self.L, self.D))
            O_den = np.zeros((self.L,1))

            for x in X:
                alphas = self.forward(x, normalize=True)
                betas = self.backward(x, normalize=True)
                M = len(x)

                for word in range(1, M):
                    temp_top = [[0. for _ in range(self.L)] for _ in range(self.L)]
                    for cur_type in range(self.L):
                        for next_type in range(self.L):
                            temp_top[cur_type][next_type] = alphas[word][cur_type]*self.A[cur_type][next_type]*self.O[next_type][x[word]]*betas[word+1][next_type]

                    norm = sum(sum(temp_top, []))

                    for cur_type in range(self.L):
                        for next_type in range(self.L):
                            temp_top[cur_type][next_type] /= norm
                            A_num[cur_type][next_type] += temp_top[cur_type][next_type]

                for word in range(1, M+1):
                    temp_down = [0. for _ in range(self.L)]

                    for norm_type in range(self.L):
                        temp_down[norm_type] = alphas[word][norm_type]*betas[word][norm_type]

                    norm = sum(temp_down)

                    for cur_type in range(self.L):
                        temp_down[cur_type] /= norm
                        O_den[cur_type] += temp_down[cur_type]
                        O_num[cur_type][x[word-1]] += temp_down[cur_type]
                        if (word != M):
                            A_den[cur_type] += temp_down[cur_type]

            self.A = np.divide(A_num,A_den)
            self.O = np.divide(O_num, O_den)

            




        pass



    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 
        Arguments:
            M:          Length of the emission to generate.
        Returns:
            emission:   The randomly generated emission as a list.
            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []
        
        # Starting state is chosen uniformly at random
        starting_state = random.choice(range(self.L))
        states.append(starting_state)

        for word in range(M):
            
            # Sample next observation.

            emission.append(np.random.choice(range(self.D), p = self.O[states[word]]))

            # Sample next state.
            states.append(np.random.choice(range(self.L), p = self.A[states[word]]))
        states = states[:-1]

        return emission, states


    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)
    
    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    random.seed(2019)
    observations = set()
    for x in X:
        observations |= set(x)
    
    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm
    
    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
