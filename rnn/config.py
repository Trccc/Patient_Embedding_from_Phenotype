from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C

# Dataset name: flowers, birds
__C.MIN_EPOCH = 5
__C.EMBEDDING_DIM = 60
__C.BATCH_SIZE = 32
__C.GPU_ID = 0
__C.CUDA = True
__C.WORKERS = 4
__C.REPEAT = 100
__C.MAX_EPOCH = 100
__C.MIN_EPOCH = 5
__C.VERBOSE = 0
__C.COLNAME = 'icd_ndc'
# Pretrain

__C.PATH = edict()
__C.PATH.WEIGHTS = '../ckd_data/pre_training/diag_pres_weights.npy'
__C.PATH.WEIGHTS_IDX = '../ckd_data/pre_training/diag_pres_word2idx.npy'
__C.PATH.BASE = '../ckd_data/shuffle/diag_pres'  #folder contains the data
 


