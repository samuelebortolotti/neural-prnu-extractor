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

def train_parsing():
	"""Summary line.

	Extended description of function.

	Args:
			arg1 (int): Description of arg1
			arg2 (str): Description of arg2

	Returns:
			bool: Description of return value

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
	argspar = parser.parse_args()
	## Normalize noise between [0, 1]
	argspar.val_noiseL /= 255.
	argspar.noiseIntL[0] /= 255.
	argspar.noiseIntL[1] /= 255.
	## wiener
	#argspar.wiener = True
	## Batch size
	argspar.batch_size = 128
	argspar.val_batch_size = 128
	## resume training
	# argspar.resume_training = True
	## log dir, has changed
	#argspar.log_dir = '/content/drive/MyDrive/SIV/prova'
	## gray
	#argspar.gray = True
	

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