""" __main__.py
Main module that parses command line arguments.

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

import argparse
import ffdnet.processors as processors

def get_args():
  r"""Parse command line arguments."""

  parser = argparse.ArgumentParser(
    prog='ffdnet photo response non-uniformity',
    description='FFDNet for image denoising',
  )

  # subparsers
  subparsers = parser.add_subparsers(help='sub-commands help')
  processors.train_ffdnet.configure_subparsers(subparsers)
  processors.test_ffdnet.configure_subparsers(subparsers)
  processors.prnu_ffdnet.configure_subparsers(subparsers)
  processors.prepare_patches.configure_subparsers(subparsers)
  processors.prepare_vision_dataset.configure_subparsers(subparsers)

  # parse arguments
  parsed_args = parser.parse_args()

  if 'func' not in parsed_args:
    parser.print_usage()
    parser.exit(1)

  if 'gray' in parsed_args and parsed_args.gray != True and 'filter' in parsed_args and parsed_args.filter == 'wiener':
    parser.exit(1, 'wiener training requires images to be gray')

  return parsed_args

def main(args):
  r"""Main function."""
  args.func(
    args,
  )

if __name__ == '__main__':
  main(get_args())