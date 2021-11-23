"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import argparse
import sys
from ffdnet.train import train
from pathlib import Path

def train_parsing():
  """
	Function used to train the FFDNet neural network.
	
	After collecting some command line argument through the ArgumentParser
	it calls the `train`  function in the `ffdnet.train.py` function

  """
  parser = argparse.ArgumentParser(description="FFDNet")
  parser.add_argument("--gray", action='store_true',\
            help='train grayscale image denoising instead of RGB')

  parser.add_argument("--log_dir", type=str, default="logs", \
           help='path of log files')
  #Training parameters
  parser.add_argument("--batch_size", type=int, default=128, 	\
           help="Training batch size")
  parser.add_argument("--epochs", "--e", type=int, default=80, \
           help="Number of total training epochs")
  parser.add_argument("--resume_training", "--r", action='store_true',\
            help="resume training from a previous checkpoint")
  parser.add_argument("--milestone", nargs=2, type=int, default=[50, 60], \
            help="When to decay learning rate; should be lower than 'epochs'")
  parser.add_argument("--lr", type=float, default=1e-3, \
           help="Initial learning rate")
  parser.add_argument("--no_orthog", action='store_true',\
            help="Don't perform orthogonalization as regularization")
  parser.add_argument("--save_every", type=int, default=10,\
            help="Number of training steps to log psnr and perform \
            orthogonalization")
  parser.add_argument("--save_every_epochs", type=int, default=5,\
            help="Number of training epochs to save state")
  parser.add_argument("--noiseIntL", nargs=2, type=int, default=[0, 75], \
           help="Noise training interval")
  parser.add_argument("--val_noiseL", type=float, default=25, \
            help='noise level used on validation set')
  parser.add_argument("--wiener", action='store_true',\
            help="Apply wiener filter to extract noise from dataset")
  parser.add_argument('--val_batch_size', type=int, default=128, 	\
           	help='Validation batch size')
  parser.add_argument('--traindbf', type=Path, default='train_rgb.h5',
						help='h5py file containing the images for training the net')
  parser.add_argument('--valdbf', type=Path, default='val_rgb.h5',
						help='h5py file containing the images for validating the net')
  argspar = parser.parse_args()
	# Normalize noise between [0, 1]
  argspar.val_noiseL /= 255.
  argspar.noiseIntL[0] /= 255.
  argspar.noiseIntL[1] /= 255.
  
  if not (argspar.traindbf).exists():
    parser.exit("The file {} for training does not exists".format(argspar.traindbf))

  if not (argspar.valdbf).exists():
    parser.exit("The file {} for training does not exists".format(argspar.valdbf))  

  print("\n### Training FFDNet model ###")
  print("> Parameters:")
  for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')
  train(argspar)


if __name__ == "__main__":
  # Parse arguments
  print(sys.argv)
  train_parsing()