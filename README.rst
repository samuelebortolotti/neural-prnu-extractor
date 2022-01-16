
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

2. Data preparation
^^^^^^^^^^^^^^^^^^^

In order to train the provided model, it is necessary first to preprare the data.

To this purpose, a set of commands has been created. It must be specified however,
that such commands work while considering the sytax of the VISION dataset.

This code does not include image datasets, however the following may be obtained from:
`VISION Dataset <https://lesc.dinfo.unifi.it/VISION/>`_

Split into train and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, you will need to split the original dataset into training and validation.

You can learn more about how to perform this operation by executing

.. code-block:: shell

   python -m ffdnet prepare_vision --help

Generally, any dataset with a similar structure (no subfolders and images with experiment_name
``<camera_model_number>_<I|V>_<resource_type>_<resource_number>.jpg``) can be
splitted by executing the following

.. code-block:: shell

   python -m ffdnet prepare_vision \
     SOURCE_DIR \
     DESTINATION_DIR \
     --train_frac 0.7

**NOTES**

* Use ``-m`` option to move file instead of copying them
* ``--train_frac`` is used to specify the proportion of elements in training/validation

Prepare the patches
~~~~~~~~~~~~~~~~~~~

At this point, you will need to prepare the dataset composed of patches by executing
*prepare_patches.py* indicating the paths to the directories containing the
training and validation datasets by passing *--trainset_dir* and
*--valset_dir*\ , respectively.

You can learn more about how to perform this operation by executing

.. code-block:: shell

   python -m ffdnet prepare_patches --help

**EXAMPLE**

To prepare a dataset of patches 44x44 with stride 20, you can execute

.. code-block:: shell

   python -m ffdnet prepare_patches \
     SOURCE_DIR \
     DESTINATION_DIR \
     --patch_size 44 \
     --stride 20

**NOTES**

* To prepare a grayscale dataset: ``python prepare_patches.py --gray``
* *--max_number_patches* can be used to set the maximum number of patches
  contained in the database

3. Training
^^^^^^^^^^^

Train a model
~~~~~~~~~~~~~

A model can be trained after having built the training and validation databases
(i.e. *train_rgb.h5* and *val_rgb.h5* for color denoising, and *train_gray.h5*
and *val_gray.h5* for grayscale denoising).
Only training on GPU is supported.

.. code-block:: shell

   python -m ffdnet train --help

**EXAMPLE**

.. code-block:: shell

   python -m ffdnet train \
     --batch_size 128 \
     --val_batch_size 128 \
     --epochs 80 \
     --filter wiener \
     --experiment_name en \
     --gray

**NOTES**

* The training process can be monitored with TensorBoard as logs get saved
  in the *experiments/experiment_name* folder
* By default, noise added at validation is set to 25 (\ *--val_noiseL* flag)
* A previous training can be resumed passing the *--resume_training* flag
* It is possible to specify a different dataset location for training (validation) with ``--traindbf`` (``--valdbf``)
* Resource can be limited by users (when using torch 1.10.0) with the option ``--gpu_fraction``

4. Testing
^^^^^^^^^^

You can learn more about the test function by calling the help of the test subparser

.. code-block:: shell

   python -m ffdnet test --help

If you want to denoise an image using a one of the pretrained models
found under the *models* folder you can execute

.. code-block:: shell

   python -m ffdnet test \
     INPUT_IMG1 INPUT_IMG2 ... INPUT_IMGK \
     models/WEIGHTS \
     DST_FOLDER

To run the algorithm on CPU instead of GPU:

.. code-block:: shell

   python -m ffdnet test \
     INPUT_IMG1 INPUT_IMG2 ... INPUT_IMGK \
     models/WEIGHTS \
     DST_FOLDER \
     --device cpu

Or just change the flags value within the Makefile and run

.. code-block:: shell

   make test

**NOTES**

* Models have been trained for values of noise in [0, 5]
* Models have been trained with the Wiener filter as denoising method

5. PRNU data preparation
^^^^^^^^^^^^^^^^^^^^^^^^

In order to evaluate the model according to PRNU, it is necessary first to preprare the data.

To this purpose, a set of commands has been created. It must be specified however,
that such commands work while considering the sytax of the VISION dataset.

This code does not include image datasets, however the following may be obtained from:
`VISION Dataset <https://lesc.dinfo.unifi.it/VISION/>`_

Split into flat and nat
~~~~~~~~~~~~~~~~~~~~~~~

To this purpose, you will need to split the original dataset into flat and nat images.
In particular, it is required a dataset structure as it follows

.. code-block:: shell
   .
   ├── flat
   │   ├── D04_I_0001.jpg
   .....
   │   └── D06_I_0149.jpg
   └── nat
       ├── D04_I_0001.jpg
      ...
       └── D06_I_0132.jpg


You can learn more about how to perform this operation by executing

.. code-block:: shell

   python -m ffdnet prepare_prnu --help

Generally, any dataset with a similar structure (no subfolders and images with experiment_name
``<camera_model_number>_<I|V>_<flat|nat>_<resource_number>.jpg``) can be
splitted by executing the following

.. code-block:: shell

   python -m ffdnet prepare_prnu \
     SOURCE_DIR

**NOTES**

* Use ``-m`` option to move file instead of copying them
* Use ``--dst`` option to specify a different destination folder

6. PRNU evaluation
^^^^^^^^^^^^^^^^^^

To evaluate a model according to the PRNU, a set of commands with various options was created.
You can learn more about how to perform this operation by executing

.. code-block:: shell

   python -m ffdnet prnu --help


The evaluation uses dataset generated as described in the previous section to evaluate a specific model.

.. code-block:: shell

   python -m ffdnet prnu \
     PREPARED_DATASET_DIR \
     models/WEIGHTS

**NOTES**

* Use ``--sigma`` option to specify a set noise value for the dataset (if not specified this is calculated for every image)
* Use ``--gray`` option if using a gray dataset
* Use ``--cut_dim`` option to specify the size of the cut of the images used for the estimation of the PRNU


ABOUT THIS FILE
===============

Copyright 2018 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification, are permitted in any medium without royalty provided the copyright notice and this notice are preserved.  This file is offered as-is, without any warranty.

ACKNOLEDGMENTS
==============

Some of the code is based on code by Yiqi Yan yanyiqinwpu@gmail.com
