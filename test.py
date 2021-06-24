import os.path as osp
import glob
from datetime import datetime
from os import mkdir
from os import path
from sys import exit

import cv2
import numpy as np
import torch
from model.unet.model import UNet
from utils.image import show_images_side2side
import matplotlib.image as mpimg


def save_predictions_as_imgs(model, device, epoch, datetime_dir, image_folder='data/inputs/tutorial/Set5/*'):
    idx = 0
    save_dir = f'data/saved/models/superresolution-cnn/{datetime_dir}'
    if not path.exists(save_dir):
        mkdir(save_dir)

    model.eval()
    for image_path in glob.glob(image_folder):
        idx += 1
        base = osp.splitext(osp.basename(image_path))[0]
        # read images
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        test_file_path = f'{save_dir}/checkpoint-epoch{epoch}_test_{base}.png'
        cv2.imwrite(test_file_path, img)
        img_norm = img * 1.0 / 255
        img_norm = torch.from_numpy(np.transpose(img_norm[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img_norm.unsqueeze(0)
        img_LR = img_LR.to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round()
        test_output_path = f'{save_dir}/checkpoint-epoch{epoch}_output_{base}.png'
        cv2.imwrite(test_output_path, output)
        print(f'Saving {idx}.) filename={base}, test_image={test_file_path}, output_image={test_output_path}')

        img1 = mpimg.imread(test_file_path)
        img2 = mpimg.imread(test_output_path)
        show_images_side2side(img1, img2)

        break


if __name__ == '__main__':
    use_best_model = False
    model_date_dir = '0624_050830'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    default_model = UNet().to(dev)

    if use_best_model:
        model_path = f'data/saved/models/models/superresolution-cnn/{model_date_dir}/model_best.pth'
        print('Model path {:s}. \nTesting...'.format(model_path))
        saved_model = torch.load(model_path)
        default_model.load_state_dict(saved_model['state_dict'])
        save_predictions_as_imgs(default_model, dev, 'best', model_date_dir)
    else:
        model_paths = glob.glob(f'data/saved/models/models/superresolution-cnn/{model_date_dir}/*')
        checkpoints = list(filter(lambda path: 'checkpoint' in path, model_paths))
        checkpoints.sort()
        for model_path in checkpoints:
            filename_idx = osp.splitext(osp.basename(model_path))[0][-3:]
            saved_model = torch.load(model_path)
            default_model.load_state_dict(saved_model['state_dict'])
            save_predictions_as_imgs(default_model, dev, filename_idx, save_d)

