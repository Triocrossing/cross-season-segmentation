from models import model_configs
from utils.segmentor import Segmentor
import utils.joint_transforms as joint_transforms
from datasets import cityscapes, dataset_configs
from utils.misc import check_mkdir, get_global_opts, rename_keys_to_match
import os
import re
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as standard_transforms
# import h5py
import math
import cv2

import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
ignore_label = 255
id_to_trainid= {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
        3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
        7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
        14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
        18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
# see below how to call this funcition


def segment_images_in_folder(network_file, img_folder, save_folder, args):

    # get current available device
    if args['use_gpu']:
        print("Using CUDA" if torch.cuda.is_available() else "Using CPU")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        print('using cpu for computing')
        device = "cpu"

    # Network and weight loading
    model_config = model_configs.PspnetCityscapesConfig()
    if 'n_classes' in args:
        print('Initializing model with %d classes' % args['n_classes'])
        net = model_config.init_network(
            n_classes=args['n_classes'],
            for_clustering=False,
            output_features=False,
            use_original_base=args['use_original_base']).to(device)
    else:
        net = model_config.init_network().to(device)

    print('load model ' + network_file)
    state_dict = torch.load(
        network_file,
        map_location=lambda storage,
        loc: storage)
    # needed since we slightly changed the structure of the network in pspnet
    state_dict = rename_keys_to_match(state_dict)
    net.load_state_dict(state_dict)
    net.eval()

    # data loading
    input_transform = model_config.input_transform
    pre_validation_transform = model_config.pre_validation_transform
    # make sure crop size and stride same as during training
    sliding_crop = joint_transforms.SlidingCropImageOnly(
        713, args['sliding_transform_step'])

    check_mkdir(save_folder)
    t0 = time.time()

    # get all file names
    filenames_ims = list()
    filenames_segs_clr = list()
    filenames_segs_gt = list()
    print('Scanning %s for images to segment.' % img_folder)
    for root, subdirs, files in os.walk(img_folder):
        filenames = [f for f in files if f.endswith(args['img_ext'])]
        if len(filenames) > 0:
            print('Found %d images in %s' % (len(filenames), root))
            seg_path = root.replace(img_folder, save_folder)
            check_mkdir(seg_path)
            filenames_ims += [os.path.join(root, f) for f in filenames]
            filenames_segs_clr += [os.path.join(seg_path, "clr",
                                            f.replace(args['img_ext'],
                                                      '.png')) for f in filenames]
            filenames_segs_gt += [os.path.join(seg_path, "gt",
                                            f.replace(args['img_ext'],
                                                      '.png')) for f in filenames]

    # Create segmentor
    if net.n_classes == 19:  # This could be the 19 cityscapes classes
        segmentor = Segmentor(
            net,
            net.n_classes,
            colorize_fcn=cityscapes.colorize_mask,
            n_slices_per_pass=args['n_slices_per_pass'])
    else:
        segmentor = Segmentor(
            net,
            net.n_classes,
            colorize_fcn=None,
            n_slices_per_pass=args['n_slices_per_pass'])

    count = 1
    print(f'len im: {len(filenames_ims)}')
    print(f'len clr: {len(filenames_segs_clr)}')
    print(f'len gt: {len(filenames_segs_gt)}')
    invDict = invDict = dict(zip(id_to_trainid.values(), id_to_trainid.keys()))
    for im_file, save_path in zip(filenames_ims, filenames_segs_clr):
        save_pathGT = filenames_segs_gt[count-1]
        tnow = time.time()
        print(
            "[%d/%d (%.1fs/%.1fs)] %s" %
            (count,
             len(filenames_ims),
                tnow -
                t0,
                (tnow -
                 t0) /
                count *
                len(filenames_ims),
                im_file))
        predictMask = segmentor.run_and_save(
                          im_file,
                          save_path,
                          pre_sliding_crop_transform=pre_validation_transform,
                          sliding_crop=sliding_crop,
                          input_transform=input_transform,
                          skip_if_seg_exists=True,
                          use_gpu=args['use_gpu'])
        # predictMask.save(save_pathGT)
        print(im_file)
        imgo = cv2.imread(im_file)
        height = imgo.shape[0]
        width = imgo.shape[1]
        # print(predictMask)

        print(f'h {height}, w {width}')
        if save_pathGT is not None:
            check_mkdir(os.path.dirname(save_pathGT))
            resized = cv2.resize(predictMask, (width,height), interpolation = cv2.INTER_NEAREST)
            newresized = np.zeros_like(resized)
            for k, v in invDict.items():
              newresized[resized == k] = v
            cv2.imwrite(save_pathGT,newresized)
        count += 1

    tend = time.time()
    print('Time: %f' % (tend - t0))

# see below how to call this funcition


def segment_images_in_folder_for_experiments(network_folder, args):
    # Predefined image sets and paths
    if len(args['img_set']) > 0:
        if args['img_set'] == 'cmu':
            dset = dataset_configs.CmuConfig()
            args['img_path'] = dset.test_im_folder
            args['img_ext'] = dset.im_file_ending
            args['save_folder_name'] = 'cmu-annotated-test-images'
        elif args['img_set'] == 'wilddash':
            dset = dataset_configs.WilddashConfig()
            args['img_path'] = dset.val_im_folder
            args['img_ext'] = dset.im_file_ending
            args['save_folder_name'] = 'wilddash_val_results'
        elif args['img_set'] == 'rc':
            dset = dataset_configs.RobotcarConfig()
            args['img_path'] = dset.test_im_folder
            args['img_ext'] = dset.im_file_ending
            args['save_folder_name'] = 'robotcar-test-results'
        elif args['img_set'] == 'cityscapes':
            dset = dataset_configs.CityscapesConfig()
            args['img_path'] = dset.val_im_folder
            args['img_ext'] = dset.im_file_ending
            args['save_folder_name'] = 'cityscapes-val-results'
        elif args['img_set'] == 'cmu-vis':
            dset = dataset_configs.CmuConfig()
            args['img_path'] = dset.vis_test_im_folder
            args['img_ext'] = 'jpg'
            args['save_folder_name'] = 'cmu-visual-test-images'
        elif args['img_set'] == 'rc-vis':
            dset = dataset_configs.RobotcarConfig()
            args['img_path'] = dset.vis_test_im_folder
            args['img_ext'] = 'jpg'
            args['save_folder_name'] = 'robotcar-visual-test-images'
        elif args['img_set'] == 'vistas':
            dset = dataset_configs.VistasConfig()
            args['img_path'] = dset.val_im_folder
            args['img_ext'] = dset.im_file_ending
            args['save_folder_name'] = 'vistas-validation'

    if len(args['network_file']) < 1:
        print("Loading best network according to specified validation metric")
        if args['validation_metric'] == 'miou':
            with open(os.path.join(network_folder, 'bestval.txt')) as f:
                best_val_dict_str = f.read()
                bestval = eval(best_val_dict_str.rstrip())

            print("Network file %s - val miou %s" %
                  (bestval['snapshot'], bestval['mean_iu']))
        elif args['validation_metric'] == 'acc':
            with open(os.path.join(network_folder, 'bestval_acc.txt')) as f:
                best_val_dict_str = f.read()
                bestval = eval(best_val_dict_str.rstrip())

            print(
                "Network file %s - val acc %s" %
                (bestval['snapshot'], bestval['acc']))

        net_to_load = bestval['snapshot'] + '.pth'
        network_file = os.path.join(network_folder, net_to_load)

    else:
        print("Loading specified network")
        slash_inds = [i for i in range(
            len(args['network_file'])) if args['network_file'].startswith('/', i)]
        network_folder = args['network_file'][:slash_inds[-1]]
        network_file = args['network_file']

    # folder should have same name as for trained network
    if args['validation_metric'] == 'miou':
        save_folder = os.path.join(network_folder, args['save_folder_name'])
    elif args['validation_metric'] == 'acc':
        save_folder = os.path.join(
            network_folder,
            args['save_folder_name'] + '_acc')

    segment_images_in_folder(network_file, args['img_path'], save_folder, args)


if __name__ == '__main__':
    global_opts = get_global_opts()

    args = {
        'use_gpu': True,
        # 'miou' (miou over classes present in validation set), 'acc'
        'validation_metric': 'miou',
        'img_set': '',  # ox-vis, cmu-vis, wilddash , ox, cmu, cityscapes overwriter img_path, img_ext and save_folder_name. Set to empty string to ignore


        # 'img_path': '/srv/tempdd/xwang/RobotCarSeason/sun/rear',
        'img_path': '/srv/tempdd/xwang/oldRobotCarSeason/overcast-reference/rear',
        # 'img_ext': '.png',
        'img_ext': '.jpg',
        # 'save_folder_name': '/srv/tempdd/xwang/RobotCarSeason/sun/css',
        'save_folder_name': '/srv/tempdd/xwang/oldRobotCarSeason/overcast-reference/css',

        # specify this if using specific weight file
        'network_file': '/udd/xwang/xwang/cross-season-segmentation/network/RC-CS-Vistas-HingeF.pth',

        'n_slices_per_pass': 1,
        'sliding_transform_step': 2 / 3.
    }
    network_folder = '/udd/xwang/xwang/cross-season-segmentation/network'

    segment_images_in_folder_for_experiments(network_folder, args)
