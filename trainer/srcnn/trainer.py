# TODO: refactor train script from trainer into root .py file
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm
from model.srcnn.metric import psnr


def train(model, dataloader, train_data, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_psnr = 0.0
    for bi, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        image_data = data[0].to(device)
        label = data[1].to(device)

        # zero grad the optimizer
        optimizer.zero_grad()
        outputs = model(image_data)
        loss = criterion(outputs, label)

        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()

        # add loss of each item (total items in a batch = batch size)
        running_loss += loss.item()
        # calculate batch psnr (once every 'batch_size' iterations)
        batch_psnr = psnr(label, outputs)
        running_psnr += batch_psnr

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(train_data) / dataloader.batch_size)
    return final_loss, final_psnr


def validate(model, dataloader, val_data, epoch, criterion, device):
    model.eval()
    running_loss = 0.0
    running_psnr = 0.0
    with torch.no_grad():
        for bi, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            image_data = data[0].to(device)
            label = data[1].to(device)

            outputs = model(image_data)
            loss = criterion(outputs, label)

            # add loss to each item (total items in a batch = batch size)
            running_loss += loss.item()
            # calculate batch psnr (once every 'batch_size' iterations)
            batch_psnr = psnr(label, outputs)
            running_psnr += batch_psnr
        outputs = outputs.cpu()
        save_image(outputs, f'data/outputs/val_sr{epoch}.png')

    final_loss = running_loss / len(dataloader.dataset)
    final_psnr = running_psnr / int(len(val_data) / dataloader.batch_size)
    return final_loss, final_psnr