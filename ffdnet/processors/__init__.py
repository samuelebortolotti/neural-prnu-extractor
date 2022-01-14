"""
Processors:
It includes all the code to organize the subparsers and 
call the functions to execute

Original authors of the FFDNet implementation:

@author Matias Tassano (matias.tassano@parisdescartes.f)

Later changes for PRNU extraction:

@author Simone Alghisi (simone.alghisi-1@studenti.unitn.it)

@author Samuele Bortolotti (samuele.bortolotti@studenti.unitn.it)

@author Massimo Rizzoli (massimo.rizzoli@studenti.unitn.it)

Università di Trento 2021
"""

from . import (
  train_ffdnet,
  test_ffdnet,
  prnu_ffdnet,
  prepare_patches,
  prepare_vision_dataset,
  prepare_prnu_vision
)