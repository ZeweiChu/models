# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Transformer model helper methods."""
"""一些Transformer模型中的辅助函数。"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf

# Very low numbers to represent -infinity. We do not actually use -Inf, since we
# want to be able to multiply these values by zero to get zero. (-Inf * 0 = NaN)
_NEG_INF_FP32 = -1e9
_NEG_INF_FP16 = np.finfo(np.float16).min


def get_position_encoding(
    length, hidden_size, min_timescale=1.0, max_timescale=1.0e4):
  """Return positional encoding.

  Calculates the position encoding as a mix of sine and cosine functions with
  geometrically increasing wavelengths.
  用逐步增加的波长用sine和cosine来计算position encoding。
  Defined and formulized in Attention is All You Need, section 3.5.
  在Attention is All You Need, section 3.5中定义。 

  Args:
    length: Sequence length.
    length: Sequence长度.
    hidden_size: Size of the
    min_timescale: Minimum scale that will be applied at each position
    max_timescale: Maximum scale that will be applied at each position

  Returns:
    Tensor with shape [length, hidden_size]
  """
  # We compute the positional encoding in float32 even if the model uses
  # float16, as many of the ops used, like log and exp, are numerically unstable
  # in float16.
  position = tf.cast(tf.range(length), tf.float32)
  # [0,1,2,3,...,length-1]
  num_timescales = hidden_size // 2
  log_timescale_increment = (
      math.log(float(max_timescale) / float(min_timescale)) /
      (tf.cast(num_timescales, tf.float32) - 1))
  # log(10000) / 64 = 1/16
  inv_timescales = min_timescale * tf.exp(
      tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  # [1,2,...,64]
  # [1/16,2/16,...,4]
  # [e ^ (1/16), e ^ (2/16),...,e ^ 4]
  # [e ^ (1/16), e ^ (2/16),...,e ^ 4]
  scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
  # length * (hidden_size // 2)
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
  # length * hidden_size
  return signal


def get_decoder_self_attention_bias(length, dtype=tf.float32):
  """Calculate bias for decoder that maintains model's autoregressive property.
  计算decoder的bias

  Creates a tensor that masks out locations that correspond to illegal
  connections, so prediction at position i cannot draw information from future
  positions.
  创建一个tensor，用作decoder sequence的mask。
  这个mask可以确保当我们在decode位置i的时候无法看到i后面的位置信息。

  Args:
    length: int length of sequences in batch.
    length: batch当中sequence的长度。
    dtype: The dtype of the return value.
    dtype: 返回值的类型。

  Returns:
    float tensor of shape [1, 1, length, length]
    返回一个形状为[1,1,length, length]的tensor。
  """
  neg_inf = _NEG_INF_FP16 if dtype == tf.float16 else _NEG_INF_FP32
  with tf.name_scope("decoder_self_attention_bias"):
    valid_locs = tf.linalg.band_part(tf.ones([length, length], dtype=dtype),
                                     -1, 0)
    # https://www.tensorflow.org/api_docs/python/tf/linalg/band_part
    # [[1,0,0,0],
    #  [1,1,0,0],
    #  [1,1,1,0],
    #  [1,1,1,1]]
    valid_locs = tf.reshape(valid_locs, [1, 1, length, length])
    # 增加2个维度
    decoder_bias = neg_inf * (1.0 - valid_locs)
    # [[0,neg_inf,neg_inf,neg_inf],
    #  [0,0,neg_inf,neg_inf],
    #  [0,0,0,neg_inf],
    #  [0,0,0,0]]
    # 把这个decoder bias加到self attention上之后，然后经过Softmax，我们就确
    # 保位置i之后的单词位置不会得到任何的权重。
  return decoder_bias


def get_padding(x, padding_value=0, dtype=tf.float32):
  """Return float tensor representing the padding values in x.

  Args:
    x: int tensor with any shape
    padding_value: int value that
    dtype: The dtype of the return value.

  Returns:
    float tensor with same shape as x containing values 0 or 1.
      0 -> non-padding, 1 -> padding
    float tensor，和x形状相同，只有0或者1
      0表示不是padding，1表示padding
  """
  with tf.name_scope("padding"):
    return tf.cast(tf.equal(x, padding_value), dtype)


def get_padding_bias(x):
  """Calculate bias tensor from padding values in tensor.
  从padding值计算bias tensor。

  Bias tensor that is added to the pre-softmax multi-headed attention logits,
  which has shape [batch_size, num_heads, length, length]. The tensor is zero at
  non-padding locations, and -1e9 (negative infinity) at padding locations.
  加到softmax之前的bias tensor，形状是[batch_size, num_heads, length, length]。
  这个tensor在非padding位置上的值为0，在padding位置上的值为-1e9（负无穷）。

  Args:
    x: int tensor with shape [batch_size, length]
    x: 一个形状为[batch_size, length]的整数tensor。

  Returns:
    Attention bias tensor of shape [batch_size, 1, 1, length].
    一个形状为[batch_size, 1, 1, length]的bias tensor。
  """
  with tf.name_scope("attention_bias"):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF_FP32
    attention_bias = tf.expand_dims(
        tf.expand_dims(attention_bias, axis=1), axis=1)
  return attention_bias
