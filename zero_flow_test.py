import argparse

from PIL import Image
import cv2
import numpy as np
from numpy.linalg import inv
from path import Path
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms.functional as TF
from tqdm import tqdm
import os
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import custom_transforms
from flowutils.flowlib import flow_to_image, interp_gt_flow
from logger import AverageMeter
from losses import compute_cossim, compute_epe, multiscale_cossim
import models
from utils import *
from utils_model import fetch_model
print(torch.cuda.is_available())

epsilon = 1e-8

parser = argparse.ArgumentParser(description='Test Adversarial attacks on Optical Flow Networks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--name', dest='name', default='', required=True,
                    help='path to dataset')
parser.add_argument('--instance', dest='instance', default='', required=True,
                    help='Specifc instance')
parser.add_argument('--patch_name', dest='patch_name', default='', required=True,
                    help='patch name')
parser.add_argument('--feature_map_size', dest='feature_map_size', default=256, type=int,
                    help='Feature map sizes for visualization')
parser.add_argument('--whole_img', dest='whole_img', default=0.0, type=float,
                    help='Test whole image attack')
parser.add_argument('--compression', dest='compression', default=0.0, type=float,
                    help='Test whole image attack')
parser.add_argument('--example', dest='example', default=0, type=int,
                    help='Test whole image attack')
parser.add_argument('--fixed_loc_x', dest='fixed_loc_x', default=-1, type=int,
                    help='Test whole image attack')
parser.add_argument('--fixed_loc_y', dest='fixed_loc_y', default=-1, type=int,
                    help='Test whole image attack')
parser.add_argument('--mask_path', dest='mask_path', default='',
                    help='path to dataset')
parser.add_argument('--ignore_mask_flow', action='store_true', help='ignore flow in mask region')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
# parser.add_argument('-b', '--batch_size', default=1, type=int,
#                     metavar='N', help='mini-batch size')
parser.add_argument('--flownet', dest='flownet', type=str, default='FlowNetC', choices=['FlowNetS', 'FlowNetC', 'SpyNet', 'FlowNet2', 'PWCNet', 'Back2Future'],
                    help='flow network architecture. Options: FlowNetS | SpyNet')
# parser.add_argument('--image_size', type=int, default=384, help='the min(height, width) of the input image to network')
parser.add_argument('--patch_type', type=str, default='circle', help='patch type: circle or square')
parser.add_argument('--norotate', action='store_true', help='will display progressbar at terminal')
parser.add_argument('--true_motion', action='store_true', help='use the true motion according to static scene if intrinsics and depth are available')

activation = dict()


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def compute_feature_map(feature_map):
    feature_map_sizes = (args.feature_map_size, args.feature_map_size)
    feature_map = np.linalg.norm(feature_map, axis=0)
    feature_map = zoom(feature_map, zoom=(feature_map_sizes[-2]/feature_map.shape[-2], feature_map_sizes[-1]/feature_map.shape[-1]), order=1)
    feature_map = (feature_map - np.min(feature_map))/np.ptp(feature_map)  # Normalised [0,1]
    return feature_map


def compute_norm(feature_map):
    return np.linalg.norm(feature_map, axis=(1, 2)).mean()


def get_feature_maps():
    if args.flownet in ['FlowNetC']:
        list_of_feature_maps = [
            "conv3_1",
            "conv4",
            "conv5",
            "conv6",
            "flow6",
            "deconv5",
            "flow5",
            "deconv4",
            "flow4",
            "deconv3",
            "flow3",
            "deconv2",
            "predict"
        ]

    feature_maps = []
    avg_norms_of_feature_maps = []
    for fm in list_of_feature_maps:
        feature_map = activation[fm].clone().cpu().numpy()[0, ...]
        avg_norms_of_feature_maps.append(compute_norm(feature_map))
        feature_map = compute_feature_map(feature_map)
        feature_maps.append(feature_map)

    return feature_maps, avg_norms_of_feature_maps, list_of_feature_maps


def visualize_feature_maps(without_patch_feature_map, without_patch_norms, with_patch_feature_map, with_patch_norms, names):
    #plt.imshow(without_patch_feature_map[0].transpose(1, 2, 0))
    #plt.savefig(args.save_path / 'without_patch.jpg')
    # plt.close()
    #plt.imshow(with_patch_feature_map[0].transpose(1, 2, 0))
    #plt.savefig(args.save_path / 'with_patch.jpg')
    # plt.close()

    f, grid = plt.subplots(3, 15)
    for k in range(3*15):
        i = k // 15
        j = k % 15
        if i == 0:
            grid[i, j].set_title(names[j], fontdict=None, loc='center', color='k', fontsize=8)
            if j == 0:
                grid[i, j].imshow(without_patch_feature_map[j].transpose(1, 2, 0))
            else:
                grid[i, j].imshow(without_patch_feature_map[j], cmap='gray')
            grid[i, j].axis('off')
        elif i == 1:
            grid[i, j].set_title(without_patch_norms[j], fontdict=None, loc='center', color='k', fontsize=8)
            if j == 0:
                grid[i, j].imshow(with_patch_feature_map[j].transpose(1, 2, 0))
            else:
                grid[i, j].imshow(with_patch_feature_map[j], cmap='gray')
            grid[i, j].axis('off')
        elif i == 2:
            grid[i, j].set_title(with_patch_norms[j], fontdict=None, loc='center', color='k', fontsize=8)
            grid[i, j].axis('off')
            #grid[i, j].imshow(np.zeros_l)
    plt.axis('off')
    f.set_size_inches(8.0, 2.0)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.0, left=0.0, right=1.0, hspace=0.5, wspace=0.05)
    if args.result_filename == '':
        plt.savefig(args.save_path / 'result.jpg')
    else:
        plt.savefig(args.save_path / args.result_filename+'.jpg')
    plt.close()


def main():
    global activation
    global args
    args = parser.parse_args()
    save_path = Path(args.name)
    #args.save_path = args.save_path = save_path / args.flownet / args.instance / 'zero_flow_test'
    args.save_path = args.save_path = save_path / args.flownet / 'zero_flow_test'
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    pretrained_path = Path(args.name) / 'pretrained_models'
    patch_path = Path(args.name) / args.flownet / args.instance / 'patches' / args.patch_name
    args.result_filename = f'{args.instance.replace("/","_")}_{args.patch_name}'

    # Data loading code
    flow_loader_h, flow_loader_w = 384, 1280

    # create model
    print("=> fetching model")
    flow_net = fetch_model(pretrained_path=pretrained_path, args=args, return_feat_maps=True)

    # setup hooks
    if args.flownet in ['SpyNet', 'Back2Future', 'PWCNet']:
        pass
    elif args.flownet in ['FlowNetC']:
        flow_net.conv3_1.register_forward_hook(get_activation('conv3_1'))
        flow_net.conv4.register_forward_hook(get_activation('conv4'))
        flow_net.conv5.register_forward_hook(get_activation('conv5'))
        flow_net.conv6.register_forward_hook(get_activation('conv6'))
        flow_net.predict_flow6.register_forward_hook(get_activation('flow6'))
        flow_net.deconv5.register_forward_hook(get_activation('deconv5'))
        flow_net.predict_flow5.register_forward_hook(get_activation('flow5'))
        flow_net.deconv4.register_forward_hook(get_activation('deconv4'))
        flow_net.predict_flow4.register_forward_hook(get_activation('flow4'))
        flow_net.deconv3.register_forward_hook(get_activation('deconv3'))
        flow_net.predict_flow3.register_forward_hook(get_activation('flow3'))
        flow_net.deconv2.register_forward_hook(get_activation('deconv2'))
        flow_net.predict_flow2.register_forward_hook(get_activation('predict'))
    elif args.flownet in ['FlowNetS']:
        pass
    elif args.flownet in ['FlowNet2']:
        pass
    else:
        raise Exception('For this experiment it is necessary to use a pretrained flownet')

    flow_net = flow_net.cuda()

    cudnn.benchmark = True
    if args.whole_img == 0 and args.compression == 0:
        print("Loading patch from ", patch_path)
        patch = torch.load(patch_path)
        patch_shape = patch.shape
        if args.mask_path:
            mask_image = load_as_float(args.mask_path)
            mask_image = cv2.imresize(mask_image, (patch_shape[-1], patch_shape[-2]))/256.
            mask = np.array([mask_image.transpose(2, 0, 1)])
        else:
            if args.patch_type == 'circle':
                mask = createCircularMask(patch_shape[-2], patch_shape[-1]).astype('float32')
                mask = np.array([[mask, mask, mask]])
                pass
            elif args.patch_type == 'square':
                mask = np.ones(patch_shape)
    else:
        # add gaussian noise
        mean = 0
        var = 1
        sigma = var**0.5
        patch = np.random.normal(mean, sigma, (flow_loader_h, flow_loader_w, 3))
        patch = patch.reshape(3, flow_loader_h, flow_loader_w)
        mask = np.ones(patch.shape) * args.whole_img

    flow_net.eval()

    # set seed for reproductivity
    np.random.seed(1337)

    tgt_noise_img = torch.FloatTensor(1, 3, args.feature_map_size, args.feature_map_size).uniform_(0, 1)
    tgt_noise_img = torch.ones_like(tgt_noise_img)*0.5 + torch.FloatTensor(1, 3, args.feature_map_size, args.feature_map_size).uniform_(-0.5, 0.5)
    ref_noise_past_img = tgt_noise_img.clone()
    ref_noise_img = tgt_noise_img.clone()
    flow_boise_gt = torch.zeros_like(tgt_noise_img)

    # to cuda
    tgt_img_var = Variable(tgt_noise_img.cuda(), volatile=True)
    ref_past_img_var = Variable(ref_noise_past_img.cuda(), volatile=True)
    ref_img_var = Variable(ref_noise_img.cuda(), volatile=True)
    flow_gt_var = Variable(flow_boise_gt.cuda(), volatile=True)

    if type(flow_net).__name__ == 'Back2Future':
        flow_fwd = flow_net(ref_past_img_var, tgt_img_var, ref_img_var)
    else:
        flow_fwd, corr = flow_net(tgt_img_var, ref_img_var)
        corr = corr[-1]
        corr = corr.detach().cpu().numpy()[0, ...]

    wo_feature_maps, wo_norms, names = get_feature_maps()
    wo_norms = [compute_norm(corr)] + wo_norms
    wo_feature_maps = [compute_feature_map(corr)] + wo_feature_maps
    names = ['corr'] + names

    data_shape = tgt_noise_img.cpu().numpy().shape

    margin = 0
    random_x = args.fixed_loc_x
    random_y = args.fixed_loc_y

    if args.whole_img == 0:
        if args.patch_type == 'circle':
            patch_full, mask_full, _, random_x, random_y, _ = circle_transform(patch, mask, patch.copy(), data_shape, patch_shape, margin, norotate=args.norotate, fixed_loc=(random_x, random_y))
        elif args.patch_type == 'square':
            patch_full, mask_full, _, _, _ = square_transform(patch, mask, patch.clone(), data_shape, patch_shape, norotate=args.norotate)
        patch_full, mask_full = torch.FloatTensor(patch_full), torch.FloatTensor(mask_full)
    else:
        patch_full, mask_full = torch.FloatTensor(patch), torch.FloatTensor(mask)

    patch_full, mask_full = patch_full.cuda(), mask_full.cuda()
    patch_var, mask_var = Variable(patch_full), Variable(mask_full)

    patch_var_future = patch_var_past = patch_var
    mask_var_future = mask_var_past = mask_var

    # adverserial flow
    bt, _, h_gt, w_gt = flow_gt_var.shape
    forward_patch_flow = Variable(torch.cat((torch.zeros((bt, 2, h_gt, w_gt)), torch.ones((bt, 1, h_gt, w_gt))), 1).cuda(), volatile=True)

    adv_tgt_img_var = torch.mul((1-mask_var), tgt_img_var) + torch.mul(mask_var, patch_var)
    adv_ref_past_img_var = torch.mul((1-mask_var_past), ref_past_img_var) + torch.mul(mask_var_past, patch_var_past)
    adv_ref_img_var = torch.mul((1-mask_var_future), ref_img_var) + torch.mul(mask_var_future, patch_var_future)

    adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
    adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, -1, 1)
    adv_ref_img_var = torch.clamp(adv_ref_img_var, -1, 1)

    if type(flow_net).__name__ == 'Back2Future':
        adv_flow_fwd = flow_net(adv_ref_past_img_var, adv_tgt_img_var, adv_ref_img_var)
    else:
        adv_flow_fwd, adv_corr = flow_net(adv_tgt_img_var, adv_ref_img_var)
        adv_corr = adv_corr[-1]
        adv_corr = adv_corr.detach().cpu().numpy()[0, ...]

    w_feature_maps, w_norms, names = get_feature_maps()
    w_norms = [compute_norm(adv_corr)] + w_norms
    w_feature_maps = [compute_feature_map(adv_corr)] + w_feature_maps
    names = ['corr'] + names

    # add input image
    names = ['Input'] + names
    feature_map_sizes = (args.feature_map_size, args.feature_map_size)
    resized_img = zoom(ref_noise_img.cpu().numpy()[0, ...], zoom=(1, feature_map_sizes[-2]/ref_noise_img.shape[-2], feature_map_sizes[-1]/ref_noise_img.shape[-1]), order=1)
    wo_feature_maps = [resized_img] + wo_feature_maps
    wo_norms = ['Mean'] + [str(round(x, 2)) for x in wo_norms]
    resized_img = zoom(adv_ref_img_var.cpu().numpy()[0, ...], zoom=(1, feature_map_sizes[-2]/adv_ref_img_var.shape[-2], feature_map_sizes[-1]/adv_ref_img_var.shape[-1]), order=1)
    w_feature_maps = [resized_img] + w_feature_maps
    w_norms = ['Mean'] + [str(round(x, 2)) for x in w_norms]
    visualize_feature_maps(wo_feature_maps, wo_norms, w_feature_maps, w_norms, names)


if __name__ == '__main__':
    main()
