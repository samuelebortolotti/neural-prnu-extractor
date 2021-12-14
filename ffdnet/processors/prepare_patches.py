""" prepare_patches.py
Module which aims to prepare the patches for training and validation set

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
from typing_extensions import Required
from ffdnet.dataset import prepare_data


def configure_subparsers(subparsers):
  r"""Configure a new subparser for preparing the patches for FFDNet.
  
  Args:
    subparsers: subparser
  """

  """
  Subparser parameters
  Args:
    data_path: path containing the image dataset
    patch_size: size of the patches to extract from the images
    stride: size of stride to extract patches
    dataset_file: name of the file for the dataset
    total_samples: total number desired of samples 
    gray_mode: build the databases composed of grayscale patches
  """
  parser = subparsers.add_parser('prepare_patches', help='Prepare patches')
  parser.add_argument("--gray", "-g", action='store_true',
            help='prepare grayscale database instead of RGB [default: False]')
  parser.add_argument("data_path", metavar='SRC_DIR', type=str,
            help='Path to the image folder')
  parser.add_argument("dataset_file", metavar='DST_FILE', type=str,
            help='Name of the generated .h5 file inside of the dataset folder')

  # Preprocessing parameters
  parser.add_argument("--patch_size", "-ps", type=int, default=44,
            help="Patch size [default: 44]")
  parser.add_argument("--stride", "-s", type=int, default=20,
            help="Size of stride [default: 20]")
  parser.add_argument("--total_samples", "-ts", type=int, default=None,
            help="Total number of samples (distributed across all images) [default: None]")
  parser.set_defaults(func=main)

def main(args):
  r"""Checks the command line arguments and then runs prepare data.

  Args:
    args: command line arguments
  """

  print("\n### Building databases ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')

  prepare_data(
    data_path=args.data_path,
    patch_size=args.patch_size,
    stride=args.stride,
    dataset_file=args.dataset_file,
    total_samples=args.total_samples,
    gray_mode=args.gray
  )