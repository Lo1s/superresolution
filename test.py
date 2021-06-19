from os import mkdir

import torch
import cv2
import numpy as np
import glob as glob
import os

from torchvision.utils import save_image
from model.srcnn.model import SRCNNModel
from datetime import datetime
from os import path


now = datetime.now()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SRCNNModel().to(device)
saved_model = torch.load('data/saved/models/models/superresolution-cnn/0619_131607/model_best.pth')
model.load_state_dict(saved_model['state_dict'])

image_paths = glob.glob('data/inputs/tutorial/bicubic_2x/*')
for image_path in image_paths:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    test_image_name = image_path.split(os.path.sep)[-1].split('.')[0]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(image.shape[0], image.shape[1], 1)

    datetime_dir = f'data/outputs/{now.strftime("%Y%m%d%H%M%S")}'
    if not path.exists(datetime_dir):
        mkdir(datetime_dir)
    cv2.imwrite(f'{datetime_dir}/test_{test_image_name}.png', image)
    image = image / 255  # normalize the pixel value

    model.eval()
    with torch.no_grad():
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        outputs = model(image)

    outputs = outputs.cpu()
    save_image(outputs, f'{datetime_dir}/output_{test_image_name}.png')
    outputs = outputs.detach().numpy()
    outputs = outputs.reshape(outputs.shape[2], outputs.shape[3], outputs.shape[1])
    print(outputs.shape)