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

  The function loads the weigths of the FFDNet and then produces the following images 
  in the specified `args.output` folder.

  The images the method provides are:
  * wiener_denoised: image produced by applying `estimate_noise` as denoising algorithm on the green channel
  * prediction_denoised: image produced by denoising the original image using the estimated noise of the FFNDNet
  * wiener_noise: noise pattern detected by `estimate_noise`
  * prediction_noise: noise pattern predicted by the FFDNet

  Args:
    args: command line arguments
  """

  device = torch.device('cuda' if args['cuda'] else 'cpu')

  assert os.path.isfile(args['input']), 'The input file does not exists'

  if args['output'][-1] == '/' and len(args['output']) > 1:
    args['output'] = args['output'][:-1]


  image = cv2.imread(args['input'])
  image_green = image[:, :, 1]

  # Set the image equal to the cropped one
  image_green = image_green
  image_green = image_green/256
  image_green = np.expand_dims(image_green, 0)

  ## Load weigths
  in_ch = 1
  device_ids = [0]
  net = FFDNet(num_input_channels = in_ch)
  net.apply(weights_init_kaiming)
  model = nn.DataParallel(net, device_ids = device_ids).to(device)
  resumef = args['weight_path']
  checkpoint = torch.load(resumef, map_location=torch.device(device))
  model.load_state_dict(checkpoint)

  # estimate noise
  filtered, stdn = estimate_noise([image_green], (5, 5))
  filtered = torch.as_tensor(filtered, dtype=torch.float)
  filtered = Variable(filtered.to(device))

  # noise image
  imgn = torch.as_tensor(np.asarray([image_green]))
  imgn = Variable(imgn.to(device))

  # standard deviation
  stdn = Variable(torch.FloatTensor(stdn).to(device))

  # noise
  noise = torch.clamp((imgn - filtered), 0., 1.)
  noise = Variable(noise.to(device))

  # prediction
  model.eval()
  prediction = model(imgn, stdn)
  prediction = prediction.to(device)
  filtered = filtered.to(device)

  denoised_wiener = image[:, :, :]
  denoised_wiener[:, : , 1] = filtered

  denoised_ffdnet = image[:, :, :]
  denoised_ffdnet[:, : , 1] = imgn - prediction.detach()

  matplotlib.image.imsave('{}/wiener_denoised.jpg'.format(args['output']), denoised_wiener)
  matplotlib.image.imsave('{}/prediction_denoised.jpg'.format(args['output']), denoised_ffdnet)

  print('Wiener filter noise: ', np.min(np.asarray(noise.detach())), np.max(np.asarray(noise.detach())))
  print('FFDNET prediction noise', np.min(np.asarray(prediction.detach())), np.max(np.asarray(prediction.detach())))

  matplotlib.image.imsave('{}/wiener_noise.jpg'.format(args['output']), noise.squeeze().detach())
  matplotlib.image.imsave('{}/prediction_noise.jpg'.format(args['output']), prediction.squeeze().detach())

  criterion = init_loss().to(device)
  loss = compute_loss(criterion, prediction, noise, imgn).to(device)
  print('Loss function', loss.item())
  

if __name__ == "__main__":
  # Parse arguments
  parser = argparse.ArgumentParser(description="FFDNet_Test")
  parser.add_argument('--add_noise', type=str, default="True")
  parser.add_argument("--input", type=str, default="", \
            help='path to input image', required=True)
  parser.add_argument("--weight_path", type=str, default="", \
            help='path to the weights of the ffdnet', required=True)
  parser.add_argument("--suffix", type=str, default="", \
            help='suffix to add to output name')
  parser.add_argument("--noise_sigma", type=float, default=25, \
            help='noise level used on test set')
  parser.add_argument("--dont_save_results", action='store_true', \
            help="don't save output images")
  parser.add_argument("--no_gpu", action='store_true', \
            help="run model on CPU")
  parser.add_argument("--output", type=str, default="", \
            help='path to the output folder', required=True)
  argspar = parser.parse_args()
  # Normalize noises ot [0, 1]
  argspar.noise_sigma /= 255.
  # String to bool
  argspar.add_noise = (argspar.add_noise.lower() == 'true')

  # use CUDA?
  argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

  #argspar.cuda = False

  print("\n### Testing FFDNet model ###")
  print("> Parameters:")
  for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')

  test_ffdnet(**vars(argspar))