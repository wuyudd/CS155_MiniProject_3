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
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)] # (M+1) * L
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)] # (M+1) * L

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2A)
        
        # start:
        for i in range(self.L):
            probs[1][i] = self.A_start[i] * self.O[i][x[0]]
            
        # Viterbi
        for i in range(1, M): # x_j-1
            for j in range(self.L): # y_j
                max_prob = 0
                max_pre = ''
                for k in range(self.L): # y_j-1
                    # P(y_1:j-1, x_1:j-1) * A(y_j-1, a) * O(a, x_j)
                    curr_prob = probs[i][k] * self.A[k][j] * self.O[j][x[i]] 
                    if curr_prob >= max_prob: # max prob
                        max_prob = curr_prob
                        max_pre = str(k)
                probs[i+1][j] = max_prob
                seqs[i+1][j] = max_pre
        
        max_M_prob = 0
        max_M_pre = ''
        # get max_prob with len(x) = M
        for j in range(self.L):
            curr_M_prob = probs[M][j]
            if curr_M_prob >= max_M_prob:
                max_M_prob = curr_M_prob
                max_M_pre = str(j)
        
        max_rseq = max_M_pre
        # get prefix from end to start
        for l in range(M+1, 1, -1):
            ind = int(max_rseq[-1]) # object state for the last
            max_rseq += seqs[l-1][ind]
        max_seq = max_rseq[::-1] # reverse for the final max_seq
        ###
        ###
        ###
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
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)] # (M+1) * L

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2Bi)
        
        # start
        for j in range(self.L):
            alphas[1][j] = self.A_start[j] * self.O[j][x[0]]
        
        # calculate alpha
        for i in range(1, M): # x_j-1
            for j in range(self.L): # y_j
                alpha = 0
                for k in range(self.L): # y_j-1
                    alpha += alphas[i][k] * self.A[k][j] * self.O[j][x[i]]
                alphas[i+1][j] = alpha
            if normalize:
                sum_i = sum(alphas[i+1])
                for l in range(self.L):
                    alphas[i+1][l] /= sum_i 

#         if normalize: # 1:M
#             for i in range(M):
#                 sum_i = sum(alphas[i+1])
#                 for l in range(self.L):
#                     alphas[i+1][l] /= sum_i 
        ###
        ###
        ###

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
         # start
        for j in range(self.L):
            betas[-1][j] = 1
        
        # calculate beta
        for i in range(-1, -(M+1), -1): # x_j+1
            for j in range(self.L): # y_j
                beta = 0
                for k in range(self.L): # y_j+1
                    if i == -M:
                        beta += betas[i][k] * self.A_start[k] * self.O[k][x[i]] # A_start
                    else:
                        beta += betas[i][k] * self.A[j][k] * self.O[k][x[i]]
                betas[i-1][j] = beta
        
            # normalize:
            if normalize:
                sum_i = sum(betas[i-1])
                for l in range(self.L):
                    betas[i-1][l] /= sum_i 
#         if normalize:
#             for i in range(0, -(M+1), -1): # -1:(-(M+1))
#                 sum_i = sum(betas[i-1])
#                 for l in range(self.L):
#                     betas[i-1][l] /= sum_i 
        ###
        ###
        ###

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

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        N = len(Y)
        A_ab = [[0. for _ in range(self.L)] for _ in range(self.L)] # stores count of y_i(j) = b and y_i(j-1) = a
        A_a = [0. for _ in range(self.L)] # stores count of y_i(j-1) = a
        
        for i in range(N):
            for j in range(1, len(Y[i])):
                A_ab[Y[i][j-1]][Y[i][j]] += 1
                A_a[Y[i][j-1]] += 1
                
        for a in range(self.L):
            for b in range(self.L):
                self.A[a][b] = A_ab[a][b] / A_a[a]
        ###
        ###
        ###

        # Calculate each element of O using the M-step formulas.
        O_wa = [[0. for _ in range(self.D)] for _ in range(self.L)]
        O_a = [0. for _ in range(self.L)]
        
        for w in range(self.D):
            for a in range(self.L):
                for i in range(N):
                    for j in range(len(X[i])):
                        if Y[i][j] == a:
                            O_a[a] += 1
                            if X[i][j] == w:
                                O_wa[a][w] += 1                    
        for w in range(self.D):
            for a in range(self.L):
                self.O[a][w] = O_wa[a][w] / O_a[a]
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2C)
        
        ###
        ###
        ###

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

        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2D)
        
        iter = 0
        while iter < N_iters: # iter for N_iters
            # initialization
            A_ab = [[0. for i in range(self.L)] for j in range(self.L)]
            A_a = [0. for i in range(self.L)]
            O_wa = [[0. for i in range(self.D)] for j in range(self.L)]
            O_a = [0. for i in range(self.L)]
            
            # E-step
            for x in X: # loop through N samples, sample i 
                length = len(x) # length of sequence of sample i
                
                alphas = self.forward(x, normalize=True) # alphas
                betas = self.backward(x, normalize=True) # betas
                
                # P(y_j=a, x)
                for j in range(1, length+1):
                    tmp_j_x = [0. for _ in range(self.L)]
                    
                    for k in range(self.L):
                        tmp_j_x[k] = alphas[j][k] * betas[j][k] 
                    
                    sum_tmp_1 = sum(tmp_j_x)
                    # print(sum_tmp_1)
                    for l in range(self.L):
                        tmp_j_x[l] /= sum_tmp_1
                        O_a[l] += tmp_j_x[l]
                        O_wa[l][x[j-1]] += tmp_j_x[l] # j-1?
                        if j != length:
                            A_a[l] += tmp_j_x[l] # y_i_j-1


                # P(y_j=a, y_j+1=b, x)
                for m in range(1, length):
                    tmp_j_jp1_x = [[0. for _ in range(self.L)] for _ in range(self.L)]
                    for a in range(self.L): # y_j
                        for b in range(self.L): # y_j+1
                            tmp_j_jp1_x[a][b] = alphas[m][a] * self.O[b][x[m]] * self.A[a][b] * betas[m+1][b] # self.O[b][x[m+1]]?
                    
                    sum_tmp_2 = sum(sum(tmp_j_jp1_x, []))
                    
                    # P(y_j=a, y_j+1=b, x)
                    for a in range(self.L):
                        for b in range(self.L):
                            tmp_j_jp1_x[a][b] /= sum_tmp_2
                            A_ab[a][b] += tmp_j_jp1_x[a][b]

            # M-step
            for a in range(self.L):
                for b in range(self.L):
                    self.A[a][b] = A_ab[a][b] / A_a[a]

            for a in range(self.L):
                for w in range(self.D):
                    self.O[a][w] = O_wa[a][w] / O_a[a]
            
            iter += 1 
        ###
        ###
        ###

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
        #random.seed(2019)
        ###
        ###
        ### 
        ### TODO: Insert Your Code Here (2F)
        starting_state = random.randrange(0, self.L, 1)
        #print("starting_state = ", starting_state)
        states.append(starting_state)
        for i in range(M):
            
            rnd_o = random.uniform(0,1)
            next_obs = 0
            while rnd_o > 0:
                rnd_o -= self.O[states[i]][next_obs]
                next_obs += 1
            next_obs -= 1
            emission.append(next_obs) 
            
            rnd_s = random.uniform(0,1)
            curr_state = states[i]
            next_state = 0
            
            while rnd_s > 0:
                rnd_s -= self.A[states[i]][next_state]
                next_state += 1
            next_state -= 1
            states.append(next_state)
        
        ###
        ###
        ###

        return emission, states[:-1]


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
    random.seed(2019)
    # Make a set of observations.
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
