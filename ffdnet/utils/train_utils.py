""" train_utils.py
It includes several utility function for the FFDNet training

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
import math
import numpy as np
import torch
import torch.nn as nn
from enum import Enum
from torch.utils.data import DataLoader
from scipy.signal.signaltools import wiener
from skimage.restoration import estimate_sigma, denoise_wavelet
from ffdnet.dataset import Dataset
from ffdnet.models import FFDNet
from torch.autograd import Variable
import os

class Estimator(Enum):
  r"""Noise estimator class
  It is basically an enumerator, used to distinguish the denoising method to employ
  """
  # Wiener enum value
  WIENER = 0

def weights_init_kaiming(lyr):
  r"""Initializes weights of the model according to the "He" initialization
  method described in "Delving deep into rectifiers: Surpassing human-level
  performance on ImageNet classification" - He, K. et al. (2015), using a
  normal distribution.
  This function is to be called by the torch.nn.Module.apply() method,
  which applies weights_init_kaiming() to every layer of the model.

  """
  classname = lyr.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
  elif classname.find('Linear') != -1:
    nn.init.kaiming_normal_(lyr.weight.data, a=0, mode='fan_in')
  elif classname.find('BatchNorm') != -1:
    lyr.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).\
      clamp_(-0.025, 0.025)
    nn.init.constant_(lyr.bias.data, 0.0)
  

def load_dataset_and_dataloader(args):
  r"""Load the datasets and the dataloaders (for both training and validation)
  according to what has been specified through the command line arguments
  Default:

  - Dataset train shuffle = True
  - Dataset validation shuffle = False
  - DataLoader train shuffle = True
  - DataLoader validation shuffle = False

  Args:
    args: command line arguments (use traindbf, valdbf, gray, batch_size)
  Returns:
    datasets: a dictionary containing the dataset, for both training and validation, 
    respectively under the key train and val
    dataloaders: a dictionary containing the dataloaders, for both training and validation, 
    respectively under the key train and val
  """

  print('> Loading dataset ...')

  dataset_train = Dataset(dbf=args.traindbf, train=True, gray_mode=args.gray, shuffle=True)
  dataset_val = Dataset(dbf=args.valdbf, train=False, gray_mode=args.gray, shuffle=False)
  loader_train = DataLoader(dataset=dataset_train, num_workers=2, batch_size=args.batch_size, shuffle=True)
  loader_val = DataLoader(dataset=dataset_val, num_workers=2, batch_size=args.val_batch_size, shuffle=False)

  print("\t# of training samples: %d\n" % int(len(dataset_train)))

  datasets = {'train': dataset_train, 'val': dataset_val}
  dataloaders = {'train': loader_train, 'val': loader_val}
  return datasets, dataloaders

def create_model(args):
  r"""Creates the model by initializing the input channel according to the args.gray flag.
  If args.gray is specified, the FFDNET will have 1 as number of input channels, otherwise 
  it will be set to 3 (RGB). Moreover, after the Neural Network is initialized, the weights 
  are set using the weights_init_kaiming function. Finally the model is moved to GPU and if 
  possible parallelized using the number of specified gpu devices

  Args:
    args: command line arguments (use gray)
  Returns:
    model: FFDNet initialized
  """

  # Check channel number
  if not args.gray:
    in_ch = 3
  else:
    in_ch = 1
  net = FFDNet(num_input_channels=in_ch)

  # Initialize model with He init
  net.apply(weights_init_kaiming)

  # Move to GPU
  device_ids = [0]
  model = nn.DataParallel(net, device_ids=device_ids).cuda()
  return model

def init_loss():
  r"""Initializes the loss function for the FFDNet (MSE)

  Returns:
    loss: loss function
  """
  return nn.MSELoss(reduction='sum')

def resume_training(args, model, optimizer):
  r"""Resumes the training if the corresponding flag is specified.
  If the resume_training flag is set to true, the function tries to recover, from the 
  checkpoint specified, the number of epoch, training and validation parameters.
  If the resume_training flag is set to false, the parameters are set to the default ones

  Args:
    args: command line arguments (use gray, experiment_name)
    model: FFDNet
    optimizer: optimizer

  Returns:
    training_params: training parameters [step, no_orthog, num_bad_epochs]
    val_params: validation parameters [step, best_loss]
    start_epoch: epoch number
  """
  if args.resume_training:
    resumef = os.path.join(args.experiment_name, 'ckpt.pth')
    if os.path.isfile(resumef):
      checkpoint = torch.load(resumef)
      print("> Resuming previous training")
      model.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      new_epoch = args.epochs
      args = checkpoint['args']
      training_params = checkpoint['training_params']
      start_epoch = training_params['start_epoch']
      val_params = checkpoint['val_params']
      args.epochs = new_epoch
      print("=> loaded checkpoint '{}' (epoch {})".format(resumef, start_epoch))
      print("=> loaded parameters :")
      print("==> checkpoint['optimizer']['param_groups']")
      print("\t{}".format(checkpoint['optimizer']['param_groups']))
      print("==> checkpoint['training_params']")
      for k in checkpoint['training_params']:
        print("\t{}, {}".format(k, checkpoint['training_params'][k]))
      print("==> checkpoint['val_params']")
      for k in checkpoint['val_params']:
        print("\t{}, {}".format(k, checkpoint['val_params'][k]))
      argpri = vars(checkpoint['args'])
      print("==> checkpoint['args']")
      for k in argpri:
        print("\t{}, {}".format(k, argpri[k]))

      args.resume_training = False
    else:
      raise Exception("Couldn't resume training with checkpoint {}".format(resumef))
  else:
    start_epoch = 0
    training_params = {}
    training_params['step'] = 0
    training_params['no_orthog'] = args.no_orthog
    training_params['num_bad_epochs'] = 0
    val_params = {}
    val_params['step'] = 0
    val_params['best_loss'] = np.inf

  return training_params, val_params, start_epoch, args

def create_input_variables(args, data):
  r"""Creates the FFDNet input variables according the denoising method specified:
    - wiener:
      the original image is denoised by appling the Wiener filter
    - if such flag is not specified, by default the approach proposed in "FFDNet: 
      Toward a Fast and Flexible Solution for CNN based Image Denoising" is applied
  
  Args:
    args: command line arguments (use gray, experiment_name, filter)
    data: image batch

  Returns:
    img: denoised images
    imgn: noisy images
    noise: noise patterns
    stdn_var: noise standard deviations (noise treated as AWGN)
  """

  if args.filter == 'default':
    img = data
    noise = torch.zeros(img.size())
    stdn = np.random.uniform(args.noiseIntL[0], args.noiseIntL[1], size=noise.size()[0])
    for nx in range(noise.size()[0]):
      sizen = noise[0, :, :, :].size()
      noise[nx, :, :, :] = torch.FloatTensor(sizen).normal_(mean=0, std=stdn[nx])
    imgn = img + noise
    img = Variable(img.cuda())
    imgn = Variable(imgn.cuda())
  elif args.filter == 'wiener':
    imgn = data
    img, stdn = estimate_noise(imgn, wiener_kernel_size = (5, 5))
    img = torch.as_tensor(img, dtype=torch.float)
    stdn = torch.FloatTensor(stdn)
    stdn = Variable(stdn.cuda())
    img = Variable(img.cuda())
    imgn = Variable(imgn.cuda())
    noise = torch.clamp(imgn - img, 0., 1.)
  stdn_var = Variable(torch.cuda.FloatTensor(stdn))
  noise = Variable(noise.cuda())

  return img, imgn, stdn_var, noise

def compute_loss(criterion, pred, noise, imgn):
  r"""Computes the loss on the batch according the criterion

  Args:
    criterion: loss criterion
    pred: noise predictions
    noise: original noise patters
    imgn: noisy images

  Returns:
    loss value: loss value
  """
  return criterion(pred, noise) / (imgn.size()[0]*2)

def get_lr(optimizer):
  r"""Returns the learning rate value

  Args:
    optimizer: optimizer

  Returns:
    lr: learning rate
  """
  for param_group in optimizer.param_groups:
    return param_group['lr']


def estimate_noise(image_list, method = Estimator.WIENER, wiener_kernel_size=(5, 5)):
  r"""Estimate noise using the wiener filter

  Args:
    image_list: list of noisy images
    method: Noise estimator [Estimator.WIENER]
    wiener_kernel_size: Kernel size of the Wiener filter [default (5, 5)]

  Returns:
    (filtered_images, noises): tuple of filtered images and associated noise
  """
  # image should be an array of arrays [0 - 1] integer value already in grayscale

  if not isinstance(method, Estimator):
    assert False, 'Unknown noise estimator, please specify one of those listed in {}'.format([i for i in Estimator.__dict__.keys() if i[:1] != '_'])

  if isinstance(image_list, torch.Tensor) or isinstance(image_list, (np.ndarray, np.generic)):
    n_images = image_list.shape[0]
    c = image_list.shape[1]
    w = image_list.shape[2]
    h = image_list.shape[3]
  elif isinstance(image_list, list):
    n_images = len(image_list)
    image = image_list[0]
    c = image.shape[0]
    w = image.shape[1]
    h = image.shape[2]
  else:
    assert False, 'not a list/tensor/ndarray'

  filtered_images = np.ndarray((n_images, c, w, h))
  noises = np.ndarray((n_images))

  for i in range(n_images):
    image = np.asarray(image_list[i]).squeeze()

    filtered = wiener(image, wiener_kernel_size)

    array_sum = np.sum(filtered)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan:
      filtered = image

    if not array_has_nan:
      noise_sigma = estimate_sigma(image)
    else:
      noise_sigma = 0
    filtered = np.expand_dims(filtered, 0)
    filtered_images[i] = filtered
    noises[i] = noise_sigma

  return (filtered_images, noises)