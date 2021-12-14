""" prepare_vision_dataset.py
Construction of the training and validation sets from the vision dataset

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
from ffdnet.utils.dataset_utils import split_dataset_with_fixed_dimension

def configure_subparsers(subparsers):
  r"""Configure a new subparser for preparing the Vision dataset for FFDNet.
  
  Args:
    subparsers: subparser
  """

  """
  Subparser parameters:

  Args:
    src: path where the VISION dataset is stored
    dst: path where to store teh training and validation set
    train_frac: training/total image ratio [default: 0.7]
    total_train_size: Exact number of elements you want in the training set [default: None]
    total_val_size: Exact number of elements you want in the validation set [default: None]
    move: if specified the images are moved instead of copying them [default: False]
  """
  parser = subparsers.add_parser('prepare_vision', help='Building the training and validation set')
  parser.add_argument("src", metavar="SRC_DIR", type=str,
            help="The path of the VISION dataset to split")
  parser.add_argument("dst", metavar="DST_DIR", type=str,
						help="Output path for training and validation")
  parser.add_argument("--train_frac", "-tf", type=float, default=0.7,
						help="The fraction of elements that will compose the training set wrt the total [default: 0.7]")
  parser.add_argument("--total_train_size", "-tts", type=float, default=None,
						help="Total number of elements in the training set [default: None]")
  parser.add_argument("--total_val_size", "-tvs", type=float, default=None,
						help="Total number of elements in the validation set [default: None]")
  parser.add_argument("--move", "-m", action="store_true",
						help="Move elements instead of copying them [default: False]")
  parser.set_defaults(func=main)

def main(args):
  r"""Checks the command line arguments and then runs prepare data.

  Args:
    args: command line arguments
  """

  print("\n### Building Training and validation ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')

  split_dataset_with_fixed_dimension(
    dataset_folder=args.src,
    destination_folder=args.dst,
    train_frac=args.train_frac,
    total_train_size=args.total_train_size,
    total_val_size=args.total_val_size,
    copy=not args.move
	)