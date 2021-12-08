# FOLDERS
VENV := venv
PROJECT_NAME := ffdnet

# PROGRAMS AND FLAGS
PYTHON := python3
PYFLAGS := 
PIP := pip
# ======= TRAIN =========
TRAIN := -m ffdnet train
TRAIN_FLAGS := --filter wiener --gray --traindbf datasets/train_gray.h5 --valdbf datasets/val_gray.h5 --gpu_fraction 0.3
# ======= TEST  =========
TEST := -m ffdnet test
TEST_FLAGS := images/lena.jpg weigths/best.pth ./images --device cuda
# ======= DOC   =========
AUTHORS := --author "Matias Tassano, Simone Alghisi, Samuele Bortolotti, Massimo Rizzoli" 
VERSION :=-r 0.1 
LANGUAGE := --language en
SPHINX_EXTENSIONS := --extensions sphinx.ext.autodoc --extensions sphinx.ext.napoleon --extensions sphinx.ext.viewcode
DOC_FOLDER := docs

## Quickstart
SPHINX_QUICKSTART := sphinx-quickstart
SPHINX_QUICKSTART_FLAGS := --sep --no-batchfile --project ffdnet $(AUTHORS) $(VERSION) $(LANGUAGE) $(SPHINX_EXTENSIONS)

# Build
BUILDER := html
SPHINX_BUILD := make $(BUILDER)
SPHINX_API_DOC := sphinx-apidoc
SPHINX_API_DOC_FLAGS := -o $(DOC_FOLDER)/source .
SPHINX_THEME = sphinx_rtd_theme
DOC_INDEX := index.html

# INDEX.rst
define INDEX

.. ffdnet documentation master file, created by
   sphinx-quickstart on Sat Nov 20 23:38:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.rst

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

endef

export INDEX

# COLORS
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
NONE := \033[0m

# COMMANDS
ECHO := echo -e
MKDIR := mkdir -p
OPEN := xdg-open
SED := sed
	
# RULES
.PHONY: help env install install-mmlab install-dev train test doc doc-layout

help:
	@$(ECHO) '$(YELLOW)Makefile help$(NONE)'
	@$(ECHO) " \
	* env 			: generates the virtual environment using venv\n \
	* install		: install the requirements listed in requirements.txt\n \
	* install-mmlab	: install the requirements for the MMlab GPU listed in requirements.mmlabgpu.txt \n \
	* install-dev		: install the development requirements listed in requirements.dev.txt\n \
	* doc-layout 		: generates the Sphinx documentation layout\n \
	* doc 			: generates the documentation (requires an existing documentation layout)\n \
	* open-doc 		: opens the documentation\n \
	* train 		: train the FFDNet\n \
	* test 		: test the FFDNet"

env:
	@$(ECHO) '$(GREEN)Creating the virtual environment..$(NONE)'
	@$(MKDIR) $(VENV)
	@$(eval PYTHON_VERSION=$(shell $(PYTHON) --version | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]' | cut -f1,2 -d'.'))
	@$(PYTHON_VERSION) -m venv $(VENV)/$(PROJECT_NAME)
	@$(ECHO) '$(GREEN)Done$(NONE)'

install:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@pip install -r requirements.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-mmlab:
	@$(ECHO) '$(GREEN)Installing requirements for MMlab GPU..$(NONE)'
	@pip install -r requirements.mmlabgpu.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

install-dev:
	@$(ECHO) '$(GREEN)Installing requirements..$(NONE)'
	@$(PIP) install -r requirements.dev.txt
	@$(ECHO) '$(GREEN)Done$(NONE)'

doc-layout:
	@$(ECHO) '$(BLUE)Generating the Sphinx layout..$(NONE)'
	$(SPHINX_QUICKSTART) $(DOC_FOLDER) $(SPHINX_QUICKSTART_FLAGS)
	@$(ECHO) "\nimport os\nimport sys\nsys.path.insert(0, os.path.abspath('../..'))" >> $(DOC_FOLDER)/source/conf.py
	@$(ECHO) "$$INDEX" > $(DOC_FOLDER)/source/index.rst
	@$(SED) -i -e "s/html_theme = 'alabaster'/html_theme = '$(SPHINX_THEME)'/g" $(DOC_FOLDER)/source/conf.py 
	@$(ECHO) '$(BLUE)Done$(NONE)'

doc:
	@$(ECHO) '$(BLUE)Generating the documentation..$(NONE)'
	$(SPHINX_API_DOC) $(SPHINX_API_DOC_FLAGS)
	cd $(DOC_FOLDER); $(SPHINX_BUILD)
	@$(ECHO) '$(BLUE)Done$(NONE)'

open-doc:
	@$(ECHO) '$(BLUE)Open documentation..$(NONE)'
	$(OPEN) $(DOC_FOLDER)/build/$(BUILDER)/$(DOC_INDEX)
	@$(ECHO) '$(BLUE)Done$(NONE)'

train:
	@$(ECHO) '$(BLUE)Training the FFDNET..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(TRAIN) $(TRAIN_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'

test:
	@$(ECHO) '$(BLUE)Testing the FFDNET..$(NONE)'
	@$(PYTHON) $(PYFLAGS) $(TEST) $(TEST_FLAGS)
	@$(ECHO) '$(BLUE)Done$(NONE)'
