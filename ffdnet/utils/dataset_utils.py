""" dataset_utils.py

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.

Later authors:
- Simone Alghisi (simone.alghisi-1@studenti.unitn.it)
- Samuele Bortolotti (samuele.bortolotti@studenti.unitn.it)
- Massimo Rizzoli (massimo.rizzoli@studenti.unitn.it)
"""
import os
from pathlib import Path
import random
from shutil import copyfile, move
import h5py
from tqdm import tqdm
import numpy as np

def split_dataset(dataset_folder: str, destination_folder: str, train_frac: float = 0.70):
  image_partition = {}
  if os.path.exists(dataset_folder):
    for image_path in Path(dataset_folder).iterdir():
      image_name = str(image_path).split('/')[-1]
      if image_name.split('.')[-1] in ['jpg']:
        name_splits = image_name.split('_')
        camera_model = name_splits[0]
        image_type = name_splits[2]
        if camera_model not in image_partition:
          image_partition[camera_model] = {}
        if image_type not in image_partition[camera_model]:
          image_partition[camera_model][image_type] = []
        image_partition[camera_model][image_type].append(str(image_path))
    if not os.path.exists(destination_folder):
      os.mkdir(destination_folder)
    if not os.path.exists(destination_folder + '/train'):
      os.mkdir(destination_folder + '/train')
    if not os.path.exists(destination_folder + '/test'):
      os.mkdir(destination_folder + '/test')
    for image_type_dict in image_partition.values():
      for image_path_list in image_type_dict.values():
        random.shuffle(image_path_list)
        list_size = len(image_path_list) - 1
        train_size = int(list_size*train_frac)
        for i, image_path in enumerate(image_path_list):
          print("Copying {}...".format(image_path))
          if i < train_size:
            print("into {}...".format(destination_folder + '/train/' + image_path.split('/')[-1]))
            copyfile(image_path, destination_folder + '/train/' + image_path.split('/')[-1])
          else:
            print("into {}...".format(destination_folder + '/test/' + image_path.split('/')[-1]))
            copyfile(image_path, destination_folder + '/test/' + image_path.split('/')[-1])

def split_dataset_with_fixed_dimension(dataset_folder: str, destination_folder: str, train_frac: float = 0.70, total_train_size: int = None, total_val_size: int = None, copy=True):
  image_partition = {}
  total_images = 0
  img_per_model = {}
  if os.path.exists(dataset_folder):
    for image_path in tqdm(Path(dataset_folder).iterdir(), desc="Scanning through VISION dataset"):
      image_name = str(image_path).split('/')[-1]
      if image_name.split('.')[-1] in ['jpg']:
        name_splits = image_name.split('_')
        camera_model = name_splits[0]
        image_type = name_splits[2]
        if camera_model not in image_partition:
          image_partition[camera_model] = {}
        if image_type not in image_partition[camera_model]:
          image_partition[camera_model][image_type] = []
        if camera_model not in img_per_model:
          img_per_model[camera_model] = {}
        if image_type not in img_per_model[camera_model]:
          img_per_model[camera_model][image_type] = 0
        image_partition[camera_model][image_type].append(str(image_path))
        img_per_model[camera_model][image_type] += 1
        total_images += 1
    if not os.path.exists(destination_folder):
      os.mkdir(destination_folder)
    if not os.path.exists(destination_folder + '/train'):
      os.mkdir(destination_folder + '/train')
    if not os.path.exists(destination_folder + '/val'):
      os.mkdir(destination_folder + '/val')
    counters = {
      'train': 0,
      'val': 0,
    }
    totals = {
      'train': total_train_size if total_train_size is not None else total_images,
      'val': total_val_size if total_val_size is not None else total_images,
    }
    for camera_model, image_type_dict in tqdm(image_partition.items(), desc="Loading camera models"):
      for image_type, image_path_list in tqdm(image_type_dict.items(), desc="Loading {}".format(camera_model)):
        random.shuffle(image_path_list)
        list_size = len(image_path_list) - 1
        train_size = int(list_size*train_frac)
        total_train_imgs_from_model = None
        if total_train_size is not None:
          total_train_imgs_from_model = round(img_per_model[camera_model][image_type]*total_train_size/total_images)
        total_val_imgs_from_model = None
        if total_val_size is not None:
          total_val_imgs_from_model = round(img_per_model[camera_model][image_type]*total_val_size/total_images)
        train_list = image_path_list[:train_size]
        val_list = image_path_list[train_size:]
        dataset_lists = {
          'train': train_list,
          'val': val_list
        }
        total_imgs_from_model = {
          'train': total_train_imgs_from_model, 
          'val': total_val_imgs_from_model
        }
        for phase in ['train', 'val']:
          for i, image_path in tqdm(enumerate(dataset_lists[phase]), desc="Working with {} for {}".format(image_type, phase)):
            if (total_imgs_from_model[phase] is not None and i >= total_imgs_from_model[phase]) or (counters[phase] >= totals[phase]):
              break
            else:
              if copy:
                copyfile(image_path, destination_folder + '/' + phase + '/' + image_path.split('/')[-1])
              else:
                move(image_path, destination_folder + '/' + phase + '/' + image_path.split('/')[-1])
              counters[phase] += 1

def split_training_into_subfolder(n_sample: int, data_folder: str, destination_folder):
  if os.path.exists(data_folder):
    for i, img_path in enumerate(Path(data_folder).iterdir()):
      img_path = str(img_path)
      destination_path = destination_folder + '/' + str(i//n_sample)
      Path(destination_path).mkdir(parents=True, exist_ok=True)
      if img_path.split('.')[-1] in ['jpg']:
        # print('moving {} into {}'.format(img_path, destination_path + '/' + img_path.split('/')[-1]))
        move(img_path, destination_path + '/' + img_path.split('/')[-1])
      else:
        i-=1

def take_randomly_from_each_folder(data_folder: str, destination_folder):
  if os.path.exists(data_folder):
    for i, folder in enumerate(Path(data_folder).iterdir()):
      if os.path.isdir(folder):
        for j, img_path in enumerate(folder.iterdir()):
          img_path = str(img_path)
          Path(destination_folder).mkdir(parents=True, exist_ok=True)
          if img_path.split('.')[-1] in ['jpg']:
            #print('copy {} into {}'.format(img_path, destination_folder + '/' + img_path.split('/')[-1]))
            copyfile(img_path, destination_folder + '/' + img_path.split('/')[-1])
          else:
            j-=1

def merge_h5py(src: str, dst: str, start_key: int = None):
  key_index = start_key
  if key_index is None:
    key_index = 0
  print('starting key index: {}'.format(key_index))
  h5f_src = h5py.File(src, 'r')
  n_patches = len(list(h5f_src.keys()))
  h5f_src.close()
  print('total patches: {}'.format(n_patches))
  for key in tqdm(range(n_patches)):
    h5f_src = h5py.File(src, 'r')
    h5f_dst = h5py.File(dst, 'a')
    h5f_dst.create_dataset(str(key_index), data=h5f_src[str(key)])
    key_index += 1
    h5f_src.close()
    h5f_dst.close()
  print('last index: {}'.format(key_index))

def take_first_n_from_h5py(src: str, dst: str, n: int = None):
  print('taking first {} elements'.format(n))
  h5f_src = h5py.File(src, 'r')
  h5f_dst = h5py.File(dst, 'a')
  for i, key in tqdm(enumerate(h5f_src.keys())):
    h5f_dst.create_dataset(str(key), data=h5f_src[str(key)])
    if i > n:
      break
  h5f_src.close()
  h5f_dst.close()

def save_only_green_channel(src, dst):
  if os.path.exists(src):
    with h5py.File(src) as h5_src:
      with h5py.File(dst, 'a') as h5_dst:
        for key in tqdm(h5_src.keys()):
          img = h5_src[key]
          green_channel = img[1, :, :]
          h5_dst.create_dataset(name=key, data=np.expand_dims(green_channel, 0))