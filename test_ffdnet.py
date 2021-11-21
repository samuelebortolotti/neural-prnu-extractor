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
import os
import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from ffdnet.models import FFDNet
from ffdnet.utils.train_utils import weights_init_kaiming, estimate_noise, compute_loss, init_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

plt.rcParams["figure.figsize"] = (10,8)


def test_ffdnet(**args):
	r"""Denoises an input image with FFDNet
	"""

	device = torch.device('cuda' if args['cuda'] else 'cpu')

	print(args)

	# Other image, which is flat
	img_original = cv2.imread(args.input)
	img = img_original[:, :, 1]

	# Crop
	# TODO remove the crop
	img_cropped = img[1500:2000, 500:1000]

	fi, ax = plt.subplots(1, 2, figsize=(20, 10))
	ax[0].imshow(np.asarray(img))
	ax[0].set_title('original')
	ax[1].imshow(img_cropped)
	ax[1].set_title('crop')
	plt.show()

	# Set the image equal to the cropped one
	img = img_cropped
	img = img/256
	img = np.expand_dims(img, 0)

	## Load weigths
	in_ch = 1
	device_ids = [0]
	net = FFDNet(num_input_channels=in_ch)
	net.apply(weights_init_kaiming)
	model = nn.DataParallel(net, device_ids=device_ids).cuda()
	resumef = args.weight_path
	checkpoint = torch.load(resumef)
	model.load_state_dict(checkpoint['state_dict'])

	# estimate noise
	filtered, stdn = estimate_noise([img], (5, 5))
	filtered = torch.as_tensor(filtered, dtype=torch.float)
	filtered = Variable(filtered.cuda())

	# noise image
	imgn = torch.as_tensor(np.asarray([img]))
	imgn = Variable(imgn.cuda())

	# standard deviation
	stdn = Variable(torch.cuda.FloatTensor(stdn))

	# noise
	noise = torch.clamp((imgn - filtered), 0., 1.)
	noise = Variable(noise.cuda())

	# prediction
	model.eval()
	prediction = model(imgn, stdn)
	prediction = prediction.to(device)
	filtered = filtered.to(device)

	fi, ax = plt.subplots(1, 2, figsize=(20, 10))

	min_value = min([np.min(np.asarray(noise.detach().to(device))), np.min(np.asarray(prediction.detach().to(device)))])
	max_value = max([np.max(np.asarray(noise.detach().to(device))), np.max(np.asarray(prediction.detach().to(device)))])

	print(np.min(np.asarray(noise.detach().to(device))), np.max(np.asarray(noise.detach().to(device))))
	print(np.min(np.asarray(prediction.detach().to(device))), np.max(np.asarray(prediction.detach().to(device))))

	colormap = plt.cm.viridis #or any other colormap
	normalize = matplotlib.colors.Normalize(vmin=min_value, vmax=max_value)

	ax[0].imshow(noise.detach().squeeze().to(device), cmap=colormap, norm=normalize)
	ax[0].set_title('wiener noise')
	ax[1].imshow(prediction.squeeze().detach().to(device), cmap=colormap, norm=normalize)
	ax[1].set_title('prediction noise')
	plt.show()

	print((imgn-noise).squeeze(0).to(device))

	criterion = init_loss().to(device)
	noise = noise.to(device)
	imgn = imgn.to(device)
	loss = compute_loss(criterion, filtered, (imgn-prediction), imgn).to(device)
	print(loss.item())
	loss = compute_loss(criterion, noise, prediction, imgn).to(device)
	print(loss.item())
	

if __name__ == "__main__":
	# Parse arguments
	parser = argparse.ArgumentParser(description="FFDNet_Test")
	parser.add_argument('--add_noise', type=str, default="True")
	parser.add_argument("--input", type=str, default="", \
						help='path to input image')
	parser.add_argument("--weight_path", type=str, default="", \
						help='path to the weights of the ffdnet')
	parser.add_argument("--suffix", type=str, default="", \
						help='suffix to add to output name')
	parser.add_argument("--noise_sigma", type=float, default=25, \
						help='noise level used on test set')
	parser.add_argument("--dont_save_results", action='store_true', \
						help="don't save output images")
	parser.add_argument("--no_gpu", action='store_true', \
						help="run model on CPU")
	argspar = parser.parse_args()
	# Normalize noises ot [0, 1]
	argspar.noise_sigma /= 255.

	# String to bool
	argspar.add_noise = (argspar.add_noise.lower() == 'true')

	# use CUDA?
	argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

	print("\n### Testing FFDNet model ###")
	print("> Parameters:")
	for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
		print('\t{}: {}'.format(p, v))
	print('\n')

	test_ffdnet(**vars(argspar))