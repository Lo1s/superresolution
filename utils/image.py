import sys
from os import listdir
from random import sample

import h5py
import numpy
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import image
from sklearn.datasets import load_sample_image
from sklearn.feature_extraction import image as sklearn_image
from skimage import transform
from tqdm import tqdm

file = h5py.File('../data/inputs/tutorial/train_mscale.h5', mode='r')
example_data = file['data'][0]
example_label = file['label'][0]

# test = example_data.reshape(example_data.shape[1], example_data.shape[2], example_data.shape[0])
# print(test.shape)
#
# plt.imshow(test)
# plt.show()


def create_patches(data_img, label_img, patch_size=(33, 33), create_random_sample=False, num_patches=1000):
    data_ext_patches = sklearn_image.extract_patches_2d(data_img, patch_size)
    label_ext_patches = sklearn_image.extract_patches_2d(label_img, patch_size)
    # print(f'Patches shape: {image_patches.shape}')
    if create_random_sample:
        random_idx = sample(range(0, data_ext_patches.shape[0]), num_patches)
        data_sample_patches = np.asarray([data_ext_patches[idx] for idx in random_idx])
        label_sample_patches = np.asarray([label_ext_patches[idx] for idx in random_idx])
        # print(f'Size of the sample patches: {sample_patches.shape}')
        return data_sample_patches, label_sample_patches
    else:
        return data_ext_patches, label_ext_patches


def save_patches_to_HDF5(filepath, data_patches, label_patches):
    f = h5py.File(filepath, 'w')
    f.create_dataset('data', data=data_patches)
    f.create_dataset('label', data=label_patches)
    f.close()


def show_patches(data, label, number_of_patches=5, reshape=False):
    for i in range(0, number_of_patches):
        rand_index = random.randrange(0, len(data) - 1)
        data_item = data[rand_index]
        label_item = label[rand_index]
        show_images_side2side(data_item, label_item, reshape=reshape)


def show_images_side2side(img1, img2, reshape=False):
    f, axarr = plt.subplots(1, 2)
    if reshape:
        img1 = img1.reshape(img1.shape[1], img1.shape[2], img1.shape[0])
        img2 = img2.reshape(img2.shape[1], img2.shape[2], img2.shape[0])
    axarr[0].imshow(img1)
    axarr[1].imshow(img2)
    plt.show()


def modcrop(img, modulo):
    # check number of channels
    if np.shape(img)[2] == 1:
        sz = np.shape(img)
        sz = np.subtract(sz, np.mod(sz, modulo))
        result = img[0:sz[0]][0:sz[1]][:]  # img = imgs(1:sz(1), 1: sz(2))
    else:
        tmpsz = np.shape(img)
        sz = tmpsz[0:2]
        sz = np.subtract(sz, np.mod(sz, modulo))
        result = img[0:sz[0]][0:sz[1]][:]  # imgs = imgs(1:sz(1), 1: sz(2),:)
    return result


# loads original images and creates smaller (default=33x33) patches from them
if __name__ == '__main__':
    data_patches = []
    label_patches = []
    num_patches = 2000
    scale = 3
    # dataset can be found on https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u
    path = '../data/inputs/tutorial/T91/'
    dir_contents = listdir(path)
    print(f'Number of files: {len(dir_contents)}')
    patches_total = num_patches * len(dir_contents)
    for i, image_file in enumerate(tqdm(dir_contents)):
        filepath = path + image_file
        image_file = image.imread(filepath)
        label_image = modcrop(image_file, scale)
        label_size = label_image.shape[0:2]
        # order 3 = bicubic interpolation
        downscale_size = (int(label_size[0] * 1/scale), int(label_size[1] * 1/scale))
        downscaled_image = transform.resize(label_image, downscale_size, order=3)
        data_image = transform.resize(downscaled_image, label_size, order=3)
        # print(f'Image ({filepath} shape: {np.shape(image_file)}')
        data_p, label_p = create_patches(data_image, label_image, create_random_sample=True, num_patches=num_patches)
        data_patches.extend(data_p)
        label_patches.extend(label_p)

    if len(data_patches) != patches_total:
        print(f'Total number of data_patches={len(data_patches)} is not equal to '
              f'num_patches={num_patches} * number of images={len(dir_contents)}')
    print(f'Total data patches size: {np.shape(label_patches)}')

    if len(label_patches) != patches_total:
        print(f'Total number of label_patches={len(label_patches)} is not equal to '
              f'num_patches={num_patches} * number of images={len(dir_contents)}')
    print(f'Total label patches size: {np.shape(label_patches)}')

    save_patches_to_HDF5('../data/inputs/tutorial/train_mscale_3ch.h5', data_patches, label_patches)

    read_saved_file = h5py.File('../data/inputs/tutorial/train_mscale_3ch.h5', mode='r')
    data = read_saved_file['data'][:]
    label = read_saved_file['label'][:]
    print(f'Saved Data patches size = {np.shape(data)} \nSaved Label patches size = {np.shape(label)}')

    show_patches(data_patches, label_patches, 10)

    sys.exit()
