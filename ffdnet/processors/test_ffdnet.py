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
from skimage import exposure
from torch.autograd import Variable
from ffdnet.models import FFDNet
from ffdnet.utils.train_utils import weights_init_kaiming, estimate_noise, compute_loss, init_loss
from ffdnet.utils.data_utils import remove_dataparallel_wrapper

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

  if not os.path.isdir(args.output):
    os.mkdir(args.output)

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

  ## Load weigths
  in_ch = 1
  device_ids = [0]
  net = FFDNet(num_input_channels = in_ch)
  net.apply(weights_init_kaiming)
  resumef = args.weight_path
  checkpoint = torch.load(resumef, map_location=torch.device(args.device))
  if args.device == 'cuda':
    model = nn.DataParallel(net, device_ids = device_ids).to(args.device)
  else:
    checkpoint = remove_dataparallel_wrapper(checkpoint)
    model = net
  model.load_state_dict(checkpoint)

  for image_path in args.input:
    if not os.path.isfile(image_path):
      print('{} is not a file'.format(image_path))
      continue

    image = Image.open(image_path)
    image = np.asarray(image, dtype=np.float32)

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

    wiener_denoised = np.uint8(wiener_denoised.cpu().detach().squeeze().numpy()*255)
    prediction_denoised = torch.clamp((imgn - prediction), 0., 1.).cpu().detach().squeeze().numpy()
    prediction_denoised = np.uint8(prediction_denoised*255)

    # save images
    folder_name = './{}/{}'.format(args.output, image_name)

    if not os.path.isdir(folder_name):
      os.mkdir(folder_name)

    image_green = np.uint8(image_green.squeeze()*255)
    image_green_saved = Image.fromarray(image_green, 'L')
    image_green_saved.save('{}/{}_original.jpg'.format(folder_name, image_name))

    wiener_denoised_img = Image.fromarray(wiener_denoised, 'L')
    wiener_denoised_img.save('{}/{}_wiener_denoised.jpg'.format(folder_name, image_name))

    prediction_denoised_img = Image.fromarray(prediction_denoised, 'L')
    prediction_denoised_img.save('{}/{}_prediction_denoised.jpg'.format(folder_name, image_name))

    # original noise
    noise = noise.cpu().detach().squeeze().numpy()
    noise_img = np.uint8(noise * 255)
    noise_img = Image.fromarray(noise_img, 'L')
    noise_img.save('{}/{}_original_wiener_noise.jpg'.format(folder_name, image_name))

    prediction = torch.clamp(prediction, 0., 1.).cpu().detach().squeeze().numpy()
    prediction_img = np.uint8(prediction * 255)
    prediction_img = Image.fromarray(prediction_img, 'L')
    prediction_img.save('{}/{}_original_prediction_noise.jpg'.format(folder_name, image_name))

    # equalized
    noise_img = exposure.equalize_hist(noise)
    noise_img = np.uint8(noise_img * 255)
    noise_img = Image.fromarray(noise_img, 'L')
    noise_img.save('{}/{}_equalized_wiener_noise.jpg'.format(folder_name, image_name))

    prediction_img = exposure.equalize_hist(prediction)
    prediction_img = np.uint8(prediction_img * 255)
    prediction_img = Image.fromarray(prediction_img, 'L')
    prediction_img.save('{}/{}_equalized_prediction_noise.jpg'.format(folder_name, image_name))