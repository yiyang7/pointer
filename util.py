# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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

"""This file contains some utility functions"""

import numpy as np
import json
from collections import defaultdict

import string
import tensorflow as tf
import time
import os
FLAGS = tf.app.flags.FLAGS

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(hps, saver, sess, ckpt_dir="train"):
  """Load checkpoint from the ckpt_dir (if unspecified, this is train dir) and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      latest_filename = "checkpoint_best" if ckpt_dir=="eval" else None
      ckpt_dir_full = os.path.join(FLAGS.log_root, ckpt_dir)
#       print("ckpt_dir: ", ckpt_dir)
      ckpt_state = tf.train.get_checkpoint_state(ckpt_dir_full, latest_filename=latest_filename)
#       print("ckpt_state: ", ckpt_state)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", ckpt_dir_full, 10)
      time.sleep(10)

# def load_embedding_from_disks(glove_filename, with_indexes=True):
#     """
#     Read a GloVe txt file. If `with_indexes=True`, we return a tuple of two dictionnaries
#     `(word_to_index_dict, index_to_embedding_array)`, otherwise we return only a direct 
#     `word_to_embedding_dict` dictionnary mapping from a string to a numpy array.
#     """
#     if with_indexes:
#         word_to_index_dict = dict()
#         index_to_embedding_array = []
#     else:
#         word_to_embedding_dict = dict()

    
#     with open(glove_filename, 'r') as glove_file:
#         for (i, line) in enumerate(glove_file):
            
#             split = line.split(' ')
            
#             word = split[0]
            
#             representation = split[1:]
#             representation = np.array(
#                 [float(val) for val in representation]
#             )
            
#             if with_indexes:
#                 if word in word_to_index_dict:
#                     print ("dup word: ", word)
#                 else:
#                     word_to_index_dict[word] = i
#                 index_to_embedding_array.append(representation)
#             else:
#                 word_to_embedding_dict[word] = representation
    
#     print ("load_embedding_from_disks representation: ", len(representation))
    
#     _WORD_NOT_FOUND = [0.01]* len(representation)  # Empty representation for unknown words.
#     if with_indexes:
#         _LAST_INDEX = i + 1
#         word_to_index_dict = defaultdict(lambda: _LAST_INDEX, word_to_index_dict)
#         print ("index_to_embedding_array: ", len(index_to_embedding_array))
#         index_to_embedding_array = np.array(index_to_embedding_array + [_WORD_NOT_FOUND])
#         print ("word_to_index_dict: ", len(word_to_index_dict))
#         print ("index_to_embedding_array: ", index_to_embedding_array.shape)
        
#         word2id = word_to_index_dict
#         id2emb = index_to_embedding_array
        
#         path = "/home/ubuntu/cs224u/processed_10_1k_mymodel/processed_combine_all/combine_all_story/"
#         stories = os.listdir(path)
#         word2id, id2emb = dict(), np.zeros((len(word_to_index_dict),50))
#         count = 0
#         for s in stories:
#             f = open(path+s,"r")
#             txt = ""
#             for i in f.readlines():
#                 txt += i 
#             for p in txt.split("\n"):
#                 p = p.translate(str.maketrans('', '', string.punctuation))
#                 for w in p.split():
#                     ind = word_to_index_dict[w]
#                     emb = index_to_embedding_array[ind]
#                     if w not in word2id:
#                         word2id[w.lower()] = count
#                         id2emb[count,:] = emb
#                         count += 1
#         id2emb = id2emb[:len(word2id)]
#         _WORD_NOT_FOUND = [0.01]* 50
#         _LAST_INDEX = len(word2id)
#         word2id = defaultdict(lambda: _LAST_INDEX, word2id)
#         id2emb = id2emb + np.array(_WORD_NOT_FOUND)
        
#         return word2id, id2emb
#     else:
#         word_to_embedding_dict = defaultdict(lambda: _WORD_NOT_FOUND)
#         return word_to_embedding_dict
    