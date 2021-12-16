""" prnu_ffdnet.py
Module to test the FFDNet on the PRNU extraction

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
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ffdnet.models import FFDNet
from ffdnet.utils.train_utils import weights_init_kaiming
from ffdnet.utils.data_utils import remove_dataparallel_wrapper
from prnu import prnu_utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def configure_subparsers(subparsers):
  r"""Configure a new subparser for testing FFDNet.
  
  Args:
    subparsers: subparser
  """

  """
  Subparser parameters:

  Args:
    dataset: path to the dataset directory which should contain the nat (natural) and the flat folders
    weight_path: path to the weights of the FFDNet
    sigma: noise level to use on the test set [defalt: None]
    mean_sigma: whether to compute teh average sigma [default: False]
    output: path to the output folder, if present the function will save the image produced [default: None]
    cut_dim: size of the cut for estimating the PRNU out of the images. [default: 512 512 3]
    device: either cuda or cpu, where to load the FFDNet and the image it has to process [default: cuda]
    gray: whether the test images are gray or not [default: False]
  """
  parser = subparsers.add_parser('prnu', help='Test the FFDNet PRNU')
  parser.add_argument("dataset", type=str, \
            help='path to the dataset directory which should contain nat and flat folders')
  parser.add_argument("weight_path", type=str, \
            help='path to the weights of the FFDNet')
  parser.add_argument("--sigma", type=float, default=None, \
            help='noise level used on dataset [default: None]')
  parser.add_argument("--mean_sigma", action='store_true', \
            help='compute the average sigma [default: False]')
  parser.add_argument("--output", type=str, default=None, \
            help="path where to save the estimated PRNU images [default: None]")
  parser.add_argument("--cut_dim", type=int, nargs=3, default=[512, 512, 3], \
            help="cut dimension for the prnu estimation (W, H, Ch) [default: 512 512 3]")
  parser.add_argument("--device", choices={'cuda', 'cpu'}, default = 'cuda', \
            help="model device [default: cuda]")
  parser.add_argument("--gray", action='store_true', \
            help='test on gray images [default: False]')
  parser.set_defaults(func = main)


def main(args):
  r"""Checks the command line arguments and then runs prnu_ffdnet.

  Args:
    args: command line arguments
  """
  # check the datasets
  if args.dataset[-1] == '/' and len(args.dataset) > 1:
    args.dataset = args.dataset[:-1]
  
  # nat dataset
  if not os.path.isdir('{}/nat'.format(args.dataset)):
    exit('No nat folder inside of {}'.format(args.dataset))
  else:
    args.nat_dataset = '{}/nat'.format(args.dataset)
  
  # flat dataset
  if not os.path.isdir('{}/flat'.format(args.dataset)):
    exit('No flat folder inside of {}'.format(args.dataset))
  else:
    args.flat_dataset = '{}/flat'.format(args.dataset)

  # device
  if args.device == 'cuda' and not torch.cuda.is_available():
    args.device = 'cpu'
    print('No GPU available')

  # cut_dim
  args.cut_dim = tuple(args.cut_dim)

  print("\n### Testing PRNU FFDNet model ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')

  prnu_ffdnet(args)

def prnu_ffdnet(args):

  ## Load weigths
  in_ch = 3
  if args.gray:
    in_ch = 1

  # load the neural network
  device_ids = [0]
  net = FFDNet(num_input_channels = in_ch)
  net.apply(weights_init_kaiming)
  resumef = args.weight_path
  checkpoint = torch.load(resumef, map_location=torch.device(args.device))
  if args.device == 'cuda':
    model = nn.DataParallel(net, device_ids = device_ids)
  else:
    checkpoint = remove_dataparallel_wrapper(checkpoint)
    model = net
  model.load_state_dict(checkpoint)

  # extract prnu
  prnus = prnu_utils.prnu_extract(args.flat_dataset, model, gray=args.gray,
                                  device=args.device, cut_dim=args.cut_dim,
                                  mean_sigma=args.mean_sigma, sigma=args.sigma)

  if args.output is not None:
    # Save the prnu images
    if not os.path.isdir(args.output):
      os.mkdir(args.output)
    for device_name, device_prnu in prnus.items():
      device_prnu = Image.fromarray(device_prnu, 'L')
      device_prnu.save('{}/prnu_{}.jpg'.format(args.output, device_name))

  # test prnu
  statistics = prnu_utils.prnu_test(args.nat_dataset, 
                                    np.asarray(list(prnus.values())), 
                                    np.asarray(list(prnus.keys())), 
                                    model, gray=args.gray, cut_dim=args.cut_dim,
                                    mean_sigma=args.mean_sigma, sigma=args.sigma)
  return statistics
