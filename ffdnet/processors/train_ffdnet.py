""" train_ffdnet
Module to train the FFDNet

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
from typing import DefaultDict
from ffdnet.train import train
from pathlib import Path
from datetime import datetime

def configure_subparsers(subparsers):
  r"""Configure a new subparser for training FFDNet.
  
  Args:
    subparsers: subparser
  """
  parser = subparsers.add_parser('train', help='Train the FFDNet')
  parser.add_argument("--gray", "-g", action="store_true",\
            help="train grayscale image denoising instead of RGB [default: False]")
  parser.add_argument("--experiment_name", "-en", type=Path, \
            default=datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), \
            help="path of log directory inside of the folder experiments [default: YYYY_MM_DD_hh_mm_ss]")
  parser.add_argument("--batch_size", "-bs", type=int, default=64, 	\
            help="Training batch size [default: 64]")
  parser.add_argument("--val_batch_size", "-vbs", type=int, default=64, 	\
           	help="Validation batch size [default: 64]")
  parser.add_argument("--epochs", "-e", type=int, default=100, \
            help="Number of total training epochs [default: 100]")
  parser.add_argument("--resume_training", "-r", action="store_true",\
            help="resume training from a previous checkpoint [default: False]")
  parser.add_argument("--lr", type=float, default=1e-3, \
            help="Initial learning rate [default: 1e-3]")
  parser.add_argument("--no_orthog", action="store_true",\
            help="Don't perform orthogonalization as regularization [default: False]")
  parser.add_argument("--save_every", type=int, default=10,\
            help="Number of training steps to log psnr and perform \
            orthogonalization [default: 10]")
  parser.add_argument("--save_every_epochs", type=int, default=5,\
            help="Number of training epochs to save state [default: 5]")
  parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], \
            help="Noise training interval [default: 0, 75]")
  parser.add_argument("--val_noiseL", type=float, default=25, \
            help="noise level used on validation set [default: 25]")
  parser.add_argument("--filter", "-f", choices=['default', 'wiener'], default='default',\
            help="Apply wiener filter or default (AWGN artifical noise) to extract noise from dataset [default: default]")
  parser.add_argument("--traindbf", "-tf", type=Path, default="datasets/train_rgb.h5",
						help="h5py file containing the images for training the net [default: 'datasets/train_rgb.h5']")
  parser.add_argument("--valdbf", "-vf", type=Path, default="datasets/val_rgb.h5",
						help="h5py file containing the images for validating the net [default: 'datasets/val_rgb.h5']")
  parser.add_argument("--gpu_fraction", "-gf", type=float, default=1.0, 	\
					  help="Set the gpu to use only a fraction of the total memory [default: 1.0]")
  parser.set_defaults(func=main)

def main(args):
  r"""Checks the arguments passed to the train function and calls it.
  
  Args:
    args: command line arguments
  """
	# Normalize noise between [0, 1]
  args.val_noiseL /= 255.
  args.noiseIntL[0] /= 255.
  args.noiseIntL[1] /= 255.

  # Add experiment_name to experiments folder
  args.experiment_name = Path("experiments") / args.experiment_name

  if not (args.traindbf).exists():
    exit("The file {} for training does not exists".format(args.traindbf))

  if not (args.valdbf).exists():
    exit("The file {} for training does not exists".format(args.valdbf))

  print("\n### Training FFDNet model ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')
  train(args)