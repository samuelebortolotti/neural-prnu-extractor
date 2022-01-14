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
from ffdnet.utils.dataset_utils import rearrange_dataset_prnu

def configure_subparsers(subparsers):
  r"""Configure a new subparser for preparing the Vision dataset for prnu estimation.
  
  Args:
    subparsers: subparser
  """

  """
  Subparser parameters:

  Args:
    src: path where the VISION dataset is stored
    dst: path where to store teh training and validation set
    move: if specified the images are moved instead of copying them [default: False]
  """
  parser = subparsers.add_parser('prepare_prnu', help='Rearranging flat and nat images for prnu estimation')
  parser.add_argument("src", metavar="SRC_DIR", type=str,
            help="The path of the VISION dataset to split")
  parser.add_argument("--dst", metavar="DST_DIR", type=str, default="datasets/prnu_dataset",
						help="Output path for flat and nat images [default: datasets/prnu_dataset]")
  parser.add_argument("--move", "-m", action="store_true",
						help="Move elements instead of copying them [default: False]")
  parser.set_defaults(func=main)

def main(args):
  r"""Checks the command line arguments and then runs prepare data.

  Args:
    args: command line arguments
  """

  print("\n### Rearranging flat and nat ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')

  rearrange_dataset_prnu(
    dataset_folder=args.src,
    destination_folder=args.dst,
    copy=not args.move
  )