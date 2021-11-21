import os
from pathlib import Path
import random
from shutil import copyfile, move
import h5py
from tqdm import tqdm

def split_dataset(dataset_folder: str, destination_folder: str, train_dim: float = 0.70):
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
        train_size = int(list_size*train_dim)
        for i, image_path in enumerate(image_path_list):
          print("Copying {}...".format(image_path))
          if i < train_size:
            print("into {}...".format(destination_folder + '/train/' + image_path.split('/')[-1]))
            copyfile(image_path, destination_folder + '/train/' + image_path.split('/')[-1])
          else:
            print("into {}...".format(destination_folder + '/test/' + image_path.split('/')[-1]))
            copyfile(image_path, destination_folder + '/test/' + image_path.split('/')[-1])

def split_dataset_with_fixed_dimension(dataset_folder: str, destination_folder: str, train_dim: float = 0.70, total_train_size: int = 60, total_val_size: int = 30):
  image_partition = {}
  total_images = 0
  img_per_model = {}
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
        if camera_model not in img_per_model:
          img_per_model[camera_model] = {}
        if image_type not in img_per_model[camera_model]:
          img_per_model[camera_model][image_type] = 0
        image_partition[camera_model][image_type].append(str(image_path))
        img_per_model[camera_model][image_type] += 1
        total_images += 1
    for camera_model, img_type_dict in img_per_model.items():
      for image_type, n_images in img_type_dict.items():
        img_per_model[camera_model][image_type] = n_images/total_images
    if not os.path.exists(destination_folder):
      os.mkdir(destination_folder)
    if not os.path.exists(destination_folder + '/train'):
      os.mkdir(destination_folder + '/train')
    if not os.path.exists(destination_folder + '/test'):
      os.mkdir(destination_folder + '/test')
    for camera_model, image_type_dict in image_partition.items():
      for image_type, image_path_list in image_type_dict.items():
        random.shuffle(image_path_list)
        list_size = len(image_path_list) - 1
        train_size = int(list_size*train_dim)
        total_train_imgs_from_model = round(img_per_model[camera_model][image_type]*total_train_size)
        total_val_imgs_from_model = round(img_per_model[camera_model][image_type]*total_val_size)
        train_list = image_path_list[:train_size]
        val_list = image_path_list[train_size:]
        for i, image_path in enumerate(train_list):
          print("Copying {}...".format(image_path))
          if i >= total_train_imgs_from_model:
            break
          else:
            print("into {}...".format(destination_folder + '/train/' + image_path.split('/')[-1]))
            copyfile(image_path, destination_folder + '/train/' + image_path.split('/')[-1])
        for i, image_path in enumerate(val_list):
          print("Copying {}...".format(image_path))
          if i >= total_val_imgs_from_model:
            break
          else:
            print("into {}...".format(destination_folder + '/test/' + image_path.split('/')[-1]))
            copyfile(image_path, destination_folder + '/test/' + image_path.split('/')[-1])

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