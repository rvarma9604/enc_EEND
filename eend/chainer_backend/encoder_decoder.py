import cupy as cp
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable

class Encoder(chainer.Chain):
    def __init__(self,
                 in_size,
                 out_size,
                 dropout=0.1
                 ):
        """Enocder model

        Args:
         in_size (int): Dimension of input feature vector
         out_size (int): Dimension of hidden states and output vector
         dropout (float): dropout ratio
        """
        super(Encoder, self).__init__()
        with self.init_scope():
            self.lstm = L.NStepLSTM(1, in_size, out_size, dropout) # one layer uniderictional lstm -> not stacked

    def forward(self, xs):
        # xs: (B, T, F)
        h_0, c_0, _ = self.lstm(None, None, xs)
        # h_0: (1, B, F)
        # c_0: (1, B, F)
        return h_0, c_0

    def __call__(self, xs):
        h_0, c_0 = self.forward(xs)
        return h_0, c_0

class Decoder(chainer.Chain):
    def __init__(self,
                 in_size,
                 out_size,
                 dropout=0.1
                 ):
        """Decoder model

        Args:
         in_size (int): Dimension of input feature vector
         out_size (int): Dimension of hidden states and output vector
         dropout (float): dropout ratio
        """
        super(Decoder, self).__init__()
        with self.init_scope():
            self.in_size = in_size
            self.lstm = L.NStepLSTM(1, in_size, out_size, dropout)
            self.linear = L.Linear(out_size, 1)    # to find p_s

    def forward(self, h_s, c_s, n_speakers=None, to_train=1):
        # h_s: (1, B, F) h_0
        # c_s: (1, B, F) c_0
        # n_speakers: (B,) number of speakers (for test set None)
        # to_train: 1 to grab S+1 speakers while training; 0 to grab S speakers if given for inference 
        batch_size = h_s.shape[1]
        
        if n_speakers:

            # zeros: (B, 1, F)
            zeros = [cp.zeros((1, self.in_size)).astype(cp.float32) for i in range(batch_size)]
            #import sys
            #print(n_speakers)
            #sys.exit()
            max_speakers = max(n_speakers).tolist()
            # max_speakers = 2
            A = cp.array([])
            
            for i in range(max_speakers + to_train):
                h_s, c_s, _ = self.lstm(h_s, c_s, zeros)
                a_s = h_s[0]
                A = F.vstack((A, a_s)) if A.size else a_s
            #P = F.sigmoid(self.linear(A))    we will use sigmoid_cross_entropy
            P = self.linear(A)

            # dimension manipulation to get
            # A: (B, n_speakers, F)
            # P: (B, n_speakers, 1)
            A = F.swapaxes(A.reshape(max_speakers + to_train, batch_size, -1), 0, 1)
            P = F.swapaxes(P.reshape(max_speakers + to_train, batch_size, -1), 0, 1)

            # strip
            A = [F.get_item(a, slice(0, n_spk)) for a, n_spk in zip(A, n_speakers)]
            P = [F.get_item(p, slice(0, n_spk + to_train)) for p, n_spk in zip(P, n_speakers)]

        else:
            # don't know number of speakers so generate a_s and p_s until p_s < 0.5
            # cannot do this batch wise like above
            # process it for each group in the batch

            # zeros: (1, 1, F)
            zeros = [cp.zeros((1, self.in_size)).astype(cp.float32)]

            A = []
            for batch in range(batch_size):
                h_b, c_b = h_s[:, batch: batch + 1, :], c_s[:, batch: batch + 1, :]

                a = p = cp.array([])
                while True:
                    h_b, c_b, _ = self.lstm(h_b, c_b, zeros)
                    a_s = h_b[0]
                    p_s = F.sigmoid(self.linear(a_s))
                    if p_s.array[0] < 0.5: 
                        break
                    a = F.vstack((a, a_s)) if a.size else a_s
                    # p = F.vstack((p, p_s)) if p.size else p_s
                a = a if a.size else cp.zeros((1, h_s.shape[2])).astype(cp.float32)
                # p = p if p.size else Variable(np.array([[0]]).astype(np.float32))
                A.append(a)
                # P.append(p)

        P = P if to_train else None
        return A, None
