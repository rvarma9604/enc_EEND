import numpy as np
import cupy as cp
import chainer
import chainer.functions as F
import chainer.links as L
from itertools import permutations
from chainer import cuda
from chainer import reporter
from eend.chainer_backend.transformer import TransformerEncoder
from eend.chainer_backend.encoder_decoder import Encoder, Decoder


"""
T: number of frames
D: dimension of embeddings
B: mini-batch size
"""


def pit_loss(pred, label, label_delay=0):
    """
    Permutation-invariant training (PIT) cross entropy loss function.

    Args:
      pred:  (T,C)-shaped pre-activation values
      label: (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      min_loss: (1,)-shape mean cross entropy
      label_perms[min_index]: permutated labels
    """
    # label permutations along the speaker axis
    label_perms = [label[..., list(p)] for p
                   in permutations(range(label.shape[-1]))]
    losses = F.stack(
        [F.sigmoid_cross_entropy(
            pred[label_delay:, ...],
            l[:len(l) - label_delay, ...]) for l in label_perms])
    xp = cuda.get_array_module(losses)
    min_loss = F.min(losses) * (len(label) - label_delay)
    min_index = cuda.to_cpu(xp.argmin(losses.data))

    return min_loss, label_perms[min_index]


def batch_pit_loss(ys, ts, label_delay=0):
    """
    PIT loss over mini-batch.

    Args:
      ys: B-length list of predictions
      ts: B-length list of labels

    Returns:
      loss: (1,)-shape mean cross entropy over mini-batch
      labels: B-length list of permuted labels
    """
    loss_w_labels = [pit_loss(y, t)
                     for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = F.sum(F.stack(losses))
    n_frames = np.sum([t.shape[0] for t in ts])
    loss = loss / n_frames
    return loss, labels


def calc_diarization_error(pred, label, label_delay=0):
    """
    Calculates diarization error stats for reporting.

    Args:
      pred (ndarray): (T,C)-shaped pre-activation values
      label (ndarray): (T,C)-shaped labels in {0,1}
      label_delay: if label_delay == 5:
           pred: 0 1 2 3 4 | 5 6 ... 99 100 |
          label: x x x x x | 0 1 ... 94  95 | 96 97 98 99 100
          calculated area: | <------------> |

    Returns:
      res: dict of diarization error stats
    """
    xp = chainer.backend.get_array_module(pred)
    label = label[:len(label) - label_delay, ...]
    decisions = F.sigmoid(pred[label_delay:, ...]).array > 0.5
    n_ref = xp.sum(label, axis=-1)
    n_sys = xp.sum(decisions, axis=-1)
    res = {}
    res['speech_scored'] = xp.sum(n_ref > 0)
    res['speech_miss'] = xp.sum(
        xp.logical_and(n_ref > 0, n_sys == 0))
    res['speech_falarm'] = xp.sum(
        xp.logical_and(n_ref == 0, n_sys > 0))
    res['speaker_scored'] = xp.sum(n_ref)
    res['speaker_miss'] = xp.sum(xp.maximum(n_ref - n_sys, 0))
    res['speaker_falarm'] = xp.sum(xp.maximum(n_sys - n_ref, 0))
    n_map = xp.sum(
        xp.logical_and(label == 1, decisions == 1),
        axis=-1)
    res['speaker_error'] = xp.sum(xp.minimum(n_ref, n_sys) - n_map)
    res['correct'] = xp.sum(label == decisions) / label.shape[1]
    res['diarization_error'] = (
        res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    return res


def report_diarization_error(ys, labels, observer):
    """
    Reports diarization errors using chainer.reporter

    Args:
      ys: B-length list of predictions (Variable)
      labels: B-length list of labels (ndarray)
      observer: target link (chainer.Chain)
    """
    for y, t in zip(ys, labels):
        stats = calc_diarization_error(y.array, t)
        for key in stats:
            reporter.report({key: stats[key]}, observer)


def probability_loss(P, n_speakers):
    """Get cross-entropy loss for the probabilities reported

    Args:
     P: (B, n_speakers + 1, 1)
     n_speakers: B-length list

    Returns:
     loss_a: (1, )-shape mean cross entropy loss over mini-batch
    """
    # l: (B, n_speakers + 1, 1)
#    loss = 0
#    for p in P:
#        p = p.T
#        l = np.ones_like(p).astype(np.int32)
#        l[0, -1] = 0
#        loss += F.sigmoid_cross_entropy(p, l)
#    return loss / len(P)
    
    # New Method
    P = F.swapaxes(F.pad_sequence(P, padding=1), 1, 2)
    L = np.ones_like(P).astype(np.int32)
    for i, n in enumerate(n_speakers):
        L[i, 0, n] = 0
    return F.sigmoid_cross_entropy(P, L)
    
     
#    l = np.ones_like(P).astype(np.int32)
#    l[:, -1] = 0
#    return F.sigmoid_cross_entropy(p, l)

def align_speaker(ys, ts):
    """Match shape as num_speaker reported can be more or less

    Args:
     ys: B-length list of predictions
     ts: B-length list of predictions

    Returns:
     ys: Aligned B-length list of predictions
     ts: Aligned B-length list of predictions
    """
    num_speakers = [max(y.shape[1], t.shape[1]) for y, t in zip(ys, ts)]
    ys = [F.pad(y, ((0, 0), (0, n_spk - y.shape[1])), 'constant', constant_values=0) for y, n_spk in zip(ys, num_speakers)]
    ts = [F.cast(F.pad(F.cast(t,'f'), ((0, 0), (0, n_spk - t.shape[1])), 'constant', constant_values=0),'i').array for t, n_spk in zip(ts, num_speakers)]
    return ys, ts


class TransformerDiarization(chainer.Chain):
    def __init__(self,
                 in_size,
                 n_units,
                 n_heads,
                 n_layers,
                 dropout,
                 alpha=1.0
                ):
        super(TransformerDiarization, self).__init__()
        with self.init_scope():
            self.enc = TransformerEncoder(
                in_size, n_layers, n_units, h=n_heads)
            self.encoder = Encoder(n_units, n_units, dropout)
            self.decoder = Decoder(n_units, n_units, dropout)
            self.alpha = alpha


    def forward(self, xs, n_speakers, activation=None):
        ilens = [x.shape[0] for x in xs]
        # xs: (B, T, F)
        xs = F.pad_sequence(xs, padding=-1)
        pad_shape = xs.shape
        # emb: (B*T, E)
        emb = self.enc(xs)
        # emb: (B, T, F)
        emb = F.separate(emb.reshape(pad_shape[0], pad_shape[1], -1), axis=0)
        emb = [F.get_item(e, slice(0, ilen)) for e, ilen in zip(emb, ilens)]
        emb2 = [cp.random.permutation(e) for e in emb]

        # get name: main-                 num_speakers=n_speakers, to_train=1
        #           validation/main-      num_speakers=n_speaker,  to_train=0
        #           validation_1/main-    num_speakers=None,       to_train=0
        name = reporter.get_current_reporter()._observer_names[id(self)]
        num_speakers = None if name=="validation_1/main" else n_speakers
        to_train = 1 if name=='main' else 0
        # h_0: (1, B, F)
        # c_0: (1, B, F)
        h_0, c_0 = self.encoder(emb2)
        # A: (B, n_spk, F)
        # P: (B, n_spk, 1)
        A, P = self.decoder(h_0, c_0, n_speakers=num_speakers, to_train=to_train)
        # yhat: (B, T, n_spk)
        ys = [F.matmul(e, a.T) for a, e in zip(A, emb)]

        return ys, P

    def estimate_sequential(self, hx, xs):
        ys, _ = self.forward(xs)
        ys = [F.sigmoid(y) for y in ys]
        return None, ys

    def __call__(self, xs, ts, n_speakers):
        # n_speakers has to be a list
        n_speakers = [n[0] for n in n_speakers]
        ys, P = self.forward(xs, n_speakers)
        # rephrase the padded ts because of padding using number of speakers
        ts = [F.get_item(t, (Ellipsis, slice(0, n))) for t, n in zip(ts, n_speakers)]

        # pad ys and ts to match num_speakers
        ys, ts = align_speaker(ys, ts)
        loss_d, labels = batch_pit_loss(ys, ts)
        # get name: main-                 num_speakers=n_speakers,   to_train=1
        #           validation/main-      num_speakers=n_speaker,    to_train=0
        #           validation_1/main-    num_speakers=None,         to_train=0
        # loss_a - for validation this loss need not be noted and we can very well return None for P and skip name retrieval
        loss_a = 0
        if P:
            loss_a = probability_loss(P, n_speakers)
            # loss_a = probability_loss(P)
        loss = loss_d + self.alpha * loss_a
        reporter.report({'loss': loss}, self)
        report_diarization_error(ys, labels, self)
        return loss

    def save_attention_weights(self, ofile, batch_index=0):
        att_weights = []
        for l in range(self.enc.n_layers):
            att_layer = getattr(self.enc, f'self_att_{l}')
            # att.shape is (B, h, T, T); pick the first sample in batch
            att_w = att_layer.att[batch_index, ...]
            att_w.to_cpu()
            att_weights.append(att_w.data)
        # save as (n_layers, h, T, T)-shaped array
        np.save(ofile, np.array(att_weights))
