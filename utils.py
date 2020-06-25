#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Dongyu Zhang
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
import six


def time_batch_generator(max_len, input_ids, labels, masks, note_ids, chunk_ids, times=None):
    """batch generator with note_id, chunk_id and time
    """
    size = len(input_ids)
    indices = np.arange(size)
    np.random.shuffle(indices)

    i = 0
    while True:
        if i < size:
            if times is not None:
                yield input_ids[indices[i]][-max_len:, :], labels[indices[i]], masks[indices[i]][-max_len:, :], \
                      note_ids[indices[i]][-max_len:], chunk_ids[indices[i]][-max_len:], times[indices[i]][-max_len:]
            else:
                yield input_ids[indices[i]][-max_len:, :], labels[indices[i]], masks[indices[i]][-max_len:, :], \
                      note_ids[indices[i]][-max_len:], chunk_ids[indices[i]][-max_len:]
            i += 1
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            continue


def mask_batch_generator(max_len, input_ids, labels, masks):
    """batch generator
    """
    size = len(input_ids)
    indices = np.arange(size)
    np.random.shuffle(indices)

    i = 0
    while True:
        if i < size:
            yield input_ids[indices[i]][-max_len:, :], labels[indices[i]], masks[indices[i]][-max_len:, :]
            i += 1
        else:
            i = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            continue

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    """Pads sequences to the same length.

    This function transforms a list of
    `num_samples` sequences (lists of integers)
    into a 2D Numpy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence otherwise.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` at the end.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.
    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.

    Pre-padding is the default.

    # Arguments
        sequences: List of lists, where each element is a sequence.
        maxlen: Int, maximum length of all sequences.
        dtype: Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, 'pre' or 'post':
            pad either before or after each sequence.
        truncating: String, 'pre' or 'post':
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value.

    # Returns
        x: Numpy array with shape `(len(sequences), maxlen)`

    # Raises
        ValueError: In case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x