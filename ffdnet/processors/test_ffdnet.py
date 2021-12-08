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
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from ffdnet.models import FFDNet
from ffdnet.utils.train_utils import weights_init_kaiming, estimate_noise, compute_loss, init_loss

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def configure_subparsers(subparsers):
  r"""Configure a new subparser for testing FFDNet.
  
  Args:
    subparsers: subparser
  """
  parser = subparsers.add_parser('test', help='Test the FFDNet')
  parser.add_argument("input", type=str, nargs="+", \
            help='input image(s)')
  parser.add_argument("weight_path", type=str, \
            help='path to the weights of the FFDNet')
  parser.add_argument("output", type=str, \
            help='path to the output folder')
  parser.add_argument("--device", choices={'cuda', 'cpu'}, default='cuda',\
            help="model device [default: cuda]")
  parser.set_defaults(func=main)

def main(args):
  r"""Checks the command line arguments and then runs test_ffdnet.

  Args:
    args: command line arguments
  """

  # device
  if args.device == 'cuda' and not torch.cuda.is_available():
    args.device = 'cpu'
    print('No GPU available')

  # output
  if args.output[-1] == '/' and len(args.output) > 1:
    args.output = args.output[:-1]

  print("\n### Testing FFDNet model ###")
  print("> Parameters:")
  for p, v in zip(args.__dict__.keys(), args.__dict__.values()):
    print('\t{}: {}'.format(p, v))
  print('\n')

  test_ffdnet(args)

def test_ffdnet(args):
  r"""Denoises an input image with FFDNet

  The function loads the weigths of the FFDNet and then produces the following images 
  in the specified `args.output` folder.

  The images the method provides are:
  * wiener_denoised: image produced by applying `estimate_noise` as denoising algorithm on the green channel
  * prediction_denoised: image produced by denoising the original image using the estimated noise of the FFNDNet

  Args:
    args: command line arguments
  """

  for image_path in args.input:
    if not os.path.isfile(image_path):
      print('{} is not a file'.format(image_path))
      continue

    image = Image.open(image_path)
    image = np.asarray(image).astype('float32')

    if image.ndim == 3 and image.shape[2] >= 2:
      image_green = image[:, :, 1]
    elif image.ndim == 3 and image.shape[2] == 1:
      image_green = image[:, :, 0]
    elif image.ndim == 2:
      image_green = image
    else:
      print('{} is not a image [ndim 2 or 3]'.format(image_path))
      continue

    # normalize
    image_green = image_green/255
    image_green = np.expand_dims(image_green, 0)

    ## Load weigths
    in_ch = 1
    device_ids = [0]
    net = FFDNet(num_input_channels = in_ch)
    net.apply(weights_init_kaiming)
    model = nn.DataParallel(net, device_ids = device_ids).to(args.device)
    resumef = args.weight_path
    checkpoint = torch.load(resumef, map_location=torch.device(args.device))
    model.load_state_dict(checkpoint)

    # estimate noise
    wiener_denoised, stdn = estimate_noise([image_green], wiener_kernel_size=(5, 5))
    wiener_denoised = torch.as_tensor(wiener_denoised, dtype=torch.float)
    wiener_denoised = Variable(wiener_denoised.to(args.device))

    # noise image
    imgn = torch.as_tensor(np.asarray([image_green]))
    imgn = Variable(imgn.to(args.device))

    # standard deviation
    stdn = Variable(torch.FloatTensor(stdn).to(args.device))

    # noise
    noise = torch.clamp((imgn - wiener_denoised), 0., 1.)

    # prediction
    model.eval()
    prediction = model(imgn, stdn)

    # loss function
    criterion = init_loss().to(args.device)
    loss = compute_loss(criterion, prediction, noise, imgn)
    print('Loss value {}: {}'.format(image_path, loss.item()))

    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # save images
    wiener_denoised = Image.fromarray(wiener_denoised.detach().squeeze().numpy(), 'L')
    wiener_denoised.save('{}/{}_wiener_denoised.jpg'.format(args.output, image_name))

    prediction_denoised = Image.fromarray((imgn - prediction).detach().squeeze().numpy(), 'L')
    prediction_denoised.save('{}/{}_prediction_denoised.jpg'.format(args.output, image_name))