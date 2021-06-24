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

model_path = 'data/saved/models/models/superresolution-cnn/0624_075046/checkpoint-epoch32.pth'
test_img_folder = 'data/inputs/tutorial/Set5/*'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = UNet().to(device)
saved_model = torch.load(model_path)
model.load_state_dict(saved_model['state_dict'])
model.eval()

print('Model path {:s}. \nTesting...'.format(model_path))

idx = 0
now = datetime.now()
datetime_dir = f'data/outputs/{now.strftime("%Y%m%d%H%M%S")}'
if not path.exists(datetime_dir):
    mkdir(datetime_dir)


for image_path in glob.glob(test_img_folder):
    idx += 1
    base = osp.splitext(osp.basename(image_path))[0]
    print(idx, base)
    # read images
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(f'{datetime_dir}/test_{base}.png', img)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()

    cv2.imwrite(f'{datetime_dir}/output_{base}.png', output)

exit()