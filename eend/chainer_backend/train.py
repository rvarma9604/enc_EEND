#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Licensed under the MIT license.

# Cupy version

import os
import cupy as cp
import numpy as np
import chainer
from chainer import optimizers
from chainer import serializers
from chainer import iterators
from chainer import training
from chainer.training import extensions
# from eend.chainer_backend.models import BLSTMDiarization
from eend.chainer_backend.models import TransformerDiarization
from eend.chainer_backend.transformer import NoamScheduler
from eend.chainer_backend.updater import GradientAccumulationUpdater
from eend.chainer_backend.diarization_dataset import KaldiDiarizationDataset
from eend.chainer_backend.utils import use_single_gpu


@chainer.dataset.converter()
def _convert(batch, device):
    def to_device_batch(batch):
        if device is None:
            return batch
        src_xp = chainer.backend.get_array_module(*batch)
        xp = device.xp
        concat = src_xp.concatenate(batch, axis=0)
        sections = list(np.cumsum(
            [len(x) for x in batch[:-1]], dtype=np.int32))
        concat_dst = device.send(concat)
        batch_dst = xp.split(concat_dst, sections)
        return batch_dst
    return {'n_speakers': to_device_batch([n for n, _, _ in batch]),
            'xs': to_device_batch([x for _, x, _ in batch]),
            'ts': to_device_batch([t for _, _, t in batch])}


def train(args):
    """ Training model with chainer backend.
    This function is called from eend/bin/train.py with
    parsed command-line arguments.
    """
    np.random.seed(args.seed)
#   cp.random.seed(args.seed)
    os.environ['CHAINER_SEED'] = str(args.seed)
    chainer.global_config.cudnn_deterministic = True

    train_set = KaldiDiarizationDataset(
        args.train_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )
    dev_set = KaldiDiarizationDataset(
        args.valid_data_dir,
        chunk_size=args.num_frames,
        context_size=args.context_size,
        input_transform=args.input_transform,
        frame_size=args.frame_size,
        frame_shift=args.frame_shift,
        subsampling=args.subsampling,
        rate=args.sampling_rate,
        use_last_samples=True,
        label_delay=args.label_delay,
        n_speakers=args.num_speakers,
        )

    # Prepare model
    n, Y, T = train_set.get_example(0)

    if args.model_type == 'Transformer':
        model = TransformerDiarization(
                in_size=Y.shape[1],
                n_units=args.hidden_size,
                n_heads=args.transformer_encoder_n_heads,
                n_layers=args.transformer_encoder_n_layers,
                dropout=args.transformer_encoder_dropout,
                alpha=args.transformer_alpha
                )
    else:
        raise ValueError('Possible model_type is "Transformer"')

    if args.gpu >= 0:
        gpuid = use_single_gpu()
        print('GPU device {} is used'.format(gpuid))
        model.to_gpu()
    else:
        gpuid = -1
    print('Prepared model')
    #gpuid = 2

    # Setup optimizer
    if args.optimizer == 'adam':
        optimizer = optimizers.Adam(alpha=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optimizers.SGD(lr=args.lr)
    elif args.optimizer == 'noam':
        optimizer = optimizers.Adam(alpha=0, beta1=0.9, beta2=0.98, eps=1e-9)
    else:
        raise ValueError(args.optimizer)

    optimizer.setup(model)
    if args.gradclip > 0:
        optimizer.add_hook(
            chainer.optimizer_hooks.GradientClipping(args.gradclip))

    # Init/Resume
    if args.initmodel: 
        serializers.load_npz(args.initmodel, model)
        print('Load model from', args.initmodel) 

    print("Done")

   # train_iter = iterators.MultiprocessIterator(
   #         train_set,
   #         batch_size=args.batchsize,
   #         repeat=True, shuffle=True,
   #         # shared_mem=64000000,
   #         shared_mem=None,
   #         n_processes=4, 
   #         n_prefetch=2,
   #         maxtasksperchild=2)

   # dev_iter = iterators.MultiprocessIterator(
   #         dev_set,
   #         batch_size=args.batchsize,
   #         repeat=False, shuffle=False,
   #         # shared_mem=64000000,
   #         shared_mem=None,
   #         n_processes=4, 
   #         n_prefetch=2,
   #         maxtasksperchild=2)
    
    train_iter = iterators.MultithreadIterator(
            train_set,
            batch_size=args.batchsize,
            repeat=True, shuffle=True,
            n_threads=10)
    dev_iter = iterators.MultithreadIterator(
            dev_set,
            batch_size=args.batchsize,
            repeat=False, shuffle=False,
            n_threads=10)

    batchs = train_iter.next()
    for (n, Y, T) in batchs:
        print(n, Y.shape, T.shape)
    
    if args.gradient_accumulation_steps > 1:
        updater = GradientAccumulationUpdater(
            train_iter, optimizer, converter=_convert, device=gpuid)
    else:
        updater = training.StandardUpdater(
            train_iter, optimizer, converter=_convert, device=gpuid)

    trainer = training.Trainer(
            updater,
            (args.max_epochs, 'epoch'),
            out=os.path.join(args.model_save_dir))

    evaluator = extensions.Evaluator(
            dev_iter, model, converter=_convert, device=gpuid)
#    evaluator_1 = extensions.Evaluator(
#            dev_iter, model, converter=_convert, device=gpuid)
    trainer.extend(evaluator)
#    trainer.extend(evaluator_1)   # to test knowing the speaker labels before hand

    if args.optimizer == 'noam':
        trainer.extend(
            NoamScheduler(args.hidden_size,
                          warmup_steps=args.noam_warmup_steps,
                          scale=args.noam_scale),
            trigger=(1, 'iteration'))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # MICRO AVERAGE
    metrics = [
            ('diarization_error', 'speaker_scored', 'DER'),
            ('speech_miss', 'speech_scored', 'SAD_MR'),
            ('speech_falarm', 'speech_scored', 'SAD_FR'),
            ('speaker_miss', 'speaker_scored', 'MI'),
            ('speaker_falarm', 'speaker_scored', 'FA'),
            ('speaker_error', 'speaker_scored', 'CF'),
            ('correct', 'frames', 'accuracy')]
    for num, den, name in metrics:
        trainer.extend(extensions.MicroAverage(
            'main/{}'.format(num),
            'main/{}'.format(den),
            'main/{}'.format(name)))
        trainer.extend(extensions.MicroAverage(
            'validation/main/{}'.format(num),
            'validation/main/{}'.format(den),
            'validation/main/{}'.format(name)))
#        trainer.extend(extensions.MicroAverage(
#            'validation_1/main/{}'.format(num),
#            'validation_1/main/{}'.format(den),
#            'validation_1/main/{}'.format(name)))

    trainer.extend(extensions.LogReport(log_name='log_iter',
                   trigger=(1000, 'iteration')))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 
         'main/DER', 'validation/main/DER',
         'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
        x_key='epoch',
        file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/DER',
         'validation/main/DER'],
        x_key='epoch',
        file_name='DER.png'))
    trainer.extend(extensions.ProgressBar(update_interval=100))
    trainer.extend(extensions.snapshot(
        filename='snapshot_epoch-{.updater.epoch}'))

    #trainer.extend(extensions.dump_graph('main/loss', out_name="cg.dot"))
    print("Started")
    trainer.run()
    print('Finished!')
