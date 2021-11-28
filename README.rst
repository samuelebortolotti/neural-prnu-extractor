
A PyTorch implementation of FFDNet image denoising.
===================================================

ABOUT
-----

Original author
^^^^^^^^^^^^^^^
   
The original FFDNET implementation was provided by


* Author  : Matias Tassano matias.tassano@parisdescartes.fr
* Copyright : (C) 2018 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+
* Source code: `http://www.ipol.im/pub/art/2019/231/ <http://www.ipol.im/pub/art/2019/231/>`_
* Reference paper: `https://doi.org/10.5201/ipol.2019.231 <https://doi.org/10.5201/ipol.2019.231>`_

Later authors
^^^^^^^^^^^^^

* `Alghisi Simone <https://github.com/Simone-Alghisi>`_\
* `Bortolotti Samuele <https://github.com/samuelebortolotti>`_\
* `Rizzoli Massimo <https://github.com/massimo-rizzoli>`_\

OVERVIEW
--------

This source code provides a PyTorch implementation of FFDNet image denoising, as in Zhang, Kai, Wangmeng Zuo, and Lei Zhang. "FFDNet: Toward a fast and flexible solution for CNN based image denoising." 

`https://arxiv.org/abs/1710.04026 <https://arxiv.org/abs/1710.04026>`_.

USER GUIDE
----------

The code as is runs in Python 3.9 with the following dependencies:

Dependencies
------------


* `PyTorch v1.10.0 <http://pytorch.org/>`_
* `scikit-image <http://scikit-image.org/>`_
* `torchvision <https://github.com/pytorch/vision>`_
* `OpenCV <https://pypi.org/project/opencv-python/>`_
* `HDF5 <http://www.h5py.org/>`_
* `tensorboard <https://github.com/tensorflow/tensorboard>`_
* `tqdm <https://github.com/tqdm/tqdm>`_

Usage
-----

0. Set up
^^^^^^^^^

.. code-block:: shell

   pip install --upgrade pip

.. code-block:: shell

   make env

.. code-block:: shell

   source venv/ffdnet/bin/activate

.. code-block:: shell

   make install

1. Documentation
^^^^^^^^^^^^^^^^

The documentation are built using `Sphinx v4.3.0 <https://www.sphinx-doc.org/en/master/>`_.

If you want to build the documentation, you need first to 
enter the project folder:

Install the development dependencies

.. code-block:: shell

   make install-dev

Build the sphinx layout

.. code-block:: shell

   make doc-layout

Build the documentation

.. code-block:: shell

   make doc

Open the documentation

.. code-block:: shell

   make open-doc

2. Testing
^^^^^^^^^^

If you want to denoise an image using a one of the pretrained models
found under the *models* folder you can execute

.. code-block::

   python test_ffdnet.py \
     --input input.png \

To run the algorithm on CPU instead of GPU:

.. code-block::

   python test_ffdnet.py \
     --input input.png \
     --no_gpu

Or simply:

.. code-block:: shell

   make test

**NOTES**


* Models have been trained for values of noise in [0, 75]

3. Training
^^^^^^^^^^^

Prepare the databases
~~~~~~~~~~~~~~~~~~~~~

First, you will need to prepare the dataset composed of patches by executing
*prepare_patches.py* indicating the paths to the directories containing the 
training and validation datasets by passing *--trainset_dir* and
*--valset_dir*\ , respectively.

Image datasets are not provided with this code, but the following can be downloaded from:
`Vision Dataset <https://lesc.dinfo.unifi.it/VISION/>`_

**NOTES**


* To prepare a grayscale dataset: ``python prepare_patches.py --gray``
* *--max_number_patches* can be used to set the maximum number of patches
  contained in the database

Train a model
~~~~~~~~~~~~~

A model can be trained after having built the training and validation databases 
(i.e. *train_rgb.h5* and *val_rgb.h5* for color denoising, and *train_gray.h5*
and *val_gray.h5* for grayscale denoising).
Only training on GPU is supported.

.. code-block::

   python train.py \
     --batch_size 128 \
     --val_batch_size 128 \
     --epochs 80 \
     --wiener \
     --experiment_name en \
     --gray

**NOTES**


* The training process can be monitored with TensorBoard as logs get saved
  in the *experiments/experiment_name* folder
* By default, noise added at validation is set to 25 (\ *--val_noiseL* flag)
* A previous training can be resumed passing the *--resume_training* flag

ABOUT THIS FILE
===============

Copyright 2018 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification, are permitted in any medium without royalty provided the copyright notice and this notice are preserved.  This file is offered as-is, without any warranty.

ACKNOLEDGMENTS
==============

Some of the code is based on code by Yiqi Yan yanyiqinwpu@gmail.com
