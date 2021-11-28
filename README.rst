
FFDNet-photo-response-non-uniformity.
===================================================

An modified version of the ``PyTorch implementation of FFDNet image denoising``, created for the ``Signal, Image, and Video`` course of the master's degree program in Artificial Intelligence System and Computer Science at the ``University of Trento``.

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

This source code provides a modified version of the "FFDNet image denoising, as in Zhang, Kai, Wangmeng Zuo, and Lei Zhang. ``FFDNet: Toward a fast and flexible solution for CNN based image denoising.``
`FFDNet paper <https://arxiv.org/abs/1710.04026>`_.

This version, unlike the original, concentrates on detecting the cameras' `PRNU <https://en.wikipedia.org/wiki/Photo_response_non-uniformity>`_.

It includes the option of training the network using the `Wiener filter <https://en.wikipedia.org/wiki/Wiener_filter>`_ as a strategy for detecting and extracting noise from images, in addition to the original method provided in the paper.

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

To facilitate the use of the application, a ``Makefile`` has been provided; to see its functions, simply call the appropriate ``help`` command with `GNU/Make <https://www.gnu.org/software/make/>`_

.. code-block:: shell

   make help

0. Set up
^^^^^^^^^

For the development phase, the Makefile provides an automatic method to create a virtual environment.

If you want to a virtual environment for the project you can run the following commands:

.. code-block:: shell

   pip install --upgrade pip

Virtual environment creation in the venv folder

.. code-block:: shell

   make env

Virtual environment activation

.. code-block:: shell

   source ./venv/ffdnet/bin/activate

Install the requirements listed in ``requirements.txt``

.. code-block:: shell

   make install

**Note:** if you have Tesla K40c GPU you can use dependency file for MMlab GPU [``requirements.mmlabgpu.txt``]

.. code-block:: shell

   make install-mmlab

1. Documentation
^^^^^^^^^^^^^^^^

The documentation is built using `Sphinx v4.3.0 <https://www.sphinx-doc.org/en/master/>`_.

If you want to build the documentation, you need first to enter the project folder:

Install the development dependencies [``requirements.dev.txt``]

.. code-block:: shell

   make install-dev

Build the Sphinx layout

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

You can learn more about the test function by calling the help of the test subparser

.. code-block:: shell

   python -m ffdnet test --help

If you want to denoise an image using a one of the pretrained models
found under the *models* folder you can execute

.. code-block:: shell

   python -m ffdnet test \
     --input input.png \
     --weight_path weigths/wiener.pth \
     --output images \
     --gray

To run the algorithm on CPU instead of GPU:

.. code-block:: shell

   python -m ffdnet test \
     --input input.png \
     --weight_path weigths/wiener.pth \
     --no_gpu \
     --output images \
     --gray

Or just change the flags value within the Makefile and run

.. code-block:: shell

   make test

**NOTES**

* Models have been trained for values of noise in [0, 75]
* Models have been trained with the Wiener filter as denoising method

3. Training
^^^^^^^^^^^

Prepare the databases
~~~~~~~~~~~~~~~~~~~~~

First, you will need to prepare the dataset composed of patches by executing
*prepare_patches.py* indicating the paths to the directories containing the 
training and validation datasets by passing *--trainset_dir* and
*--valset_dir*\ , respectively.

This code does not include image datasets, however the following may be obtained from:
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

.. code-block:: shell

   python -m ffdnet train \
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
