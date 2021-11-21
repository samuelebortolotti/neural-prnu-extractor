""" dataset.py
Dataset related functions

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import os.path
import random
import glob
import numpy as np
import cv2
import h5py
import torch
import torch.utils.data as udata
from tqdm import tqdm
from ffdnet.utils.data_utils import normalize

def img_to_patches(img, win, stride=1):
  r"""Converts an image to an array of patches.

  Args:
    img: a numpy array containing a CxHxW RGB (C=3) or grayscale (C=1)
      image
    win: size of the output patches
    stride: int. stride
  """
  k = 0
  endc = img.shape[0]
  endw = img.shape[1]
  endh = img.shape[2]
  patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
  total_pat_num = patch.shape[1] * patch.shape[2]
  res = np.zeros([endc, win*win, total_pat_num], np.float32)
  for i in range(win):
    for j in range(win):
      patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
      res[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
      k = k + 1
  return res.reshape([endc, win, win, total_pat_num])

def prepare_data(data_path, \
         val_data_path, \
         patch_size, \
         stride, \
         total_patches, \
         max_num_patches=None, \
         gray_mode=False):
  r"""Builds the training and validations datasets by scanning the
  corresponding directories for images and extracting	patches from them.

  Args:
    data_path: path containing the training image dataset
    val_data_path: path containing the validation image dataset
    patch_size: size of the patches to extract from the images
    stride: size of stride to extract patches
    max_num_patches: maximum number of patches to extract
    aug_times: number of times to augment the available data minus one
    gray_mode: build the databases composed of grayscale patches
  """
  # training database
  print('> Training database')
  types = ('*.bmp', '*.png', '*.jpg')
  files = []
  for tp in types:
    files.extend(glob.glob(os.path.join(data_path, tp)))
  files.sort()

  if gray_mode:
    traindbf = 'data/SIV/h5py/train_gray_'+ data_path.split('/')[-1] +'.h5'
    valdbf = 'data/SIV/h5py/val_gray.h5'
  else:
    traindbf = 'data/SIV/h5py/train_rgb_'+ data_path.split('/')[-1] +'.h5'
    valdbf = 'data/SIV/h5py/val_rgb.h5'

  if max_num_patches is None:
    max_num_patches = 10000000
    print("\tMaximum number of patches not set")
  else:
    print("\tMaximum number of patches set to {}".format(max_num_patches))
  train_num = 0
  val_num = 0
  train_patches = 0
  i = 0
  n_files = len(files)
  patches_per_file = total_patches//n_files
  t_file = tqdm(total=n_files)
  while i < len(files) and train_num < max_num_patches:
    with h5py.File(traindbf, 'a') as h5f:
      for j in range(5):
        if(i < len(files) and train_num < max_num_patches):
          imgor = cv2.imread(files[i])
          # h, w, c = img.shape
          img = imgor
          if not gray_mode:
            # CxHxW RGB image
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
          else:
            # CxHxW grayscale image (C=1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
          img = normalize(img)
          patches = img_to_patches(img, win=patch_size, stride=stride)
          np.random.shuffle(patches)
     
          for nx in range(patches.shape[3]):
            data = patches[:, :, :, nx] 
            train_patches += len(data)
            h5f.create_dataset(str(train_num), data=data)
            train_num += 1
            if nx-2 >= patches_per_file:
              break
          i += 1
          t_file.update(1)
  t_file.close()

  # validation database
  print('\n> Validation database')
  if val_data_path is None:
    print('\n> No')
  else:
    files = []
    for tp in types:
      files.extend(glob.glob(os.path.join(val_data_path, tp)))
    files.sort()
    h5f = h5py.File(valdbf, 'w')
    val_num = 0
    for i, item in tqdm(enumerate(files)):
      img = cv2.imread(item)
      if not gray_mode:
        # C. H. W, RGB image
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).transpose(2, 0, 1)
      else:
        # C, H, W grayscale image (C=1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, 0)
      img = normalize(img)
      h5f.create_dataset(str(val_num), data=img)
      val_num += 1
    h5f.close()

  print('\n> Total')
  print('\ttraining set, # samples %d, # patches %d' % (train_num, train_patches))
  print('\tvalidation set, # samples %d \n' % val_num)

class Dataset(udata.Dataset):
  r"""Implements torch.utils.data.Dataset
  """
  def __init__(self, train=True, gray_mode=False, shuffle=False):
    super(Dataset, self).__init__()
    self.train = train
    self.gray_mode = gray_mode
    if not self.gray_mode:
      self.traindbf = 'data/h5py/train_rgb_train_final.h5'
      self.valdbf = 'data/SIV/h5py/train_rgb_test_final.h5'
    else:
      self.traindbf = 'data/SIV/h5py/train_rgb_debug_180.h5'
      self.valdbf = 'data/SIV/h5py/val_rgb_debug_180.h5'

    if self.train:
      h5f = h5py.File(self.traindbf, 'r')
    else:
      h5f = h5py.File(self.valdbf, 'r')
    self.keys = list(h5f.keys())
    if shuffle:
      random.shuffle(self.keys)
    h5f.close()

  def __len__(self):
    return len(self.keys)

  def __getitem__(self, index):
    if self.train:
      h5f = h5py.File(self.traindbf, 'r')
    else:
      h5f = h5py.File(self.valdbf, 'r')
    key = self.keys[index]
    data = np.array(h5f[key])
    h5f.close()
    return torch.Tensor(data)