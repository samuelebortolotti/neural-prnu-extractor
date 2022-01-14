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
from tqdm import tqdm


def split_dataset_with_fixed_dimension(dataset_folder: str, destination_folder: str, train_frac: float = 0.70, \
  total_train_size: int = None, total_val_size: int = None, copy: bool = True):
  r"""splits the VISION dataset into training and validation.
  
  Args:
    dataset_folder: the original VISION dataset folder
    destination_folder: the destination folder where training and validation sets will be created
    train_frac: the fraction of elements for the training set (validation is obtained by 1-train_frac)
    total_train_size: additional parameter to fix the number of total images in the training set
    total_val_size: additional parameter to fix the number of total images in the validation set
    copy: parameter to copy images instead of cutting them
  """
  image_partition = {}
  total_images = 0
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
        image_partition[camera_model][image_type].append(str(image_path))
        total_images += 1
    if not os.path.exists(destination_folder):
      os.mkdir(destination_folder)
    if not os.path.exists(destination_folder + '/train'):
      os.mkdir(destination_folder + '/train')
    if not os.path.exists(destination_folder + '/val'):
      os.mkdir(destination_folder + '/val')
    for camera_model, image_type_dict in tqdm(image_partition.items(), desc="Loading camera models"):
      for image_type, image_path_list in tqdm(image_type_dict.items(), desc="Loading {}".format(camera_model)):
        random.shuffle(image_path_list)
        list_size = len(image_path_list)
        train_size = int(list_size*train_frac)
        train_list = image_path_list[:train_size]
        val_list = image_path_list[train_size:]
        dataset_lists = {
          'train': train_list,
          'val': val_list
        }
        total_imgs_from_model = {
          'train': round(list_size*total_train_size/total_images) if total_train_size is not None else train_size,
          'val':round(list_size*total_val_size/total_images) if total_val_size is not None else list_size-train_size
        }
        for phase in ['train', 'val']:
          total_length = min(len(dataset_lists[phase]), total_imgs_from_model[phase])
          pbar = tqdm(total_length,desc="Working with {} for {}".format(image_type, phase))
          i = 0
          while i < total_length:
            image_path = dataset_lists[phase][i]
            if copy:
              copyfile(image_path, destination_folder + '/' + phase + '/' + image_path.split('/')[-1])
            else:
              move(image_path, destination_folder + '/' + phase + '/' + image_path.split('/')[-1])
            i += 1
            pbar.update(1)
          pbar.close()

def rearrange_dataset_prnu(dataset_folder: str, destination_folder: str, copy: bool = True):
  r"""Rearrange the VISION dataset into flat and nat images for prnu estimation.
  
  Args:
    dataset_folder: the original VISION dataset folder
    destination_folder: the destination folder where to rearrange the dataset for prnu estimation
    copy: parameter to copy images instead of cutting them
  """
  if os.path.exists(dataset_folder):
    if not os.path.exists(destination_folder):
      os.mkdir(destination_folder)
    for image_path in tqdm(Path(dataset_folder).iterdir(), desc='Rearranging VISION dataset for prnu estimation'):
      image_name = str(image_path).split('/')[-1]
      if image_name.split('.')[-1] in ['jpg']:
        name_splits = image_name.split('_')
        image_type = name_splits[2]
        if not os.path.exists(destination_folder + '/' + image_type):
          os.mkdir(destination_folder + '/' + image_type)
        del name_splits[2]
        if copy:
          copyfile(image_path, destination_folder + '/' + image_type + '/' + "_".join(name_splits))
        else:
          move(image_path, destination_folder + '/' + image_type + '/' + "_".join(name_splits))
