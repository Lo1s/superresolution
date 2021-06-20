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
from tqdm import tqdm

file = h5py.File('../data/inputs/tutorial/train_mscale.h5', mode='r')
example_data = file['data'][0]
example_label = file['label'][0]

# test = example_data.reshape(example_data.shape[1], example_data.shape[2], example_data.shape[0])
# print(test.shape)
#
# plt.imshow(test)
# plt.show()


def create_patches(image, patch_size=(33, 33), create_random_sample=False, max_size=1000):
    image_patches = sklearn_image.extract_patches_2d(image, patch_size)
    # print(f'Patches shape: {image_patches.shape}')
    if create_random_sample:
        random_idx = sample(range(0, image_patches.shape[0]), max_size)
        sample_patches = np.asarray([image_patches[i] for i in random_idx])
        # print(f'Size of the sample patches: {sample_patches.shape}')
        return sample_patches
    else:
        return image_patches


def save_patches_to_HDF5(filepath, data_patches, label_patches):
    f = h5py.File(filepath, 'w')
    f.create_dataset('data', data=data_patches)
    f.create_dataset('label', data=label_patches)
    f.close()


def show_patches(data, label, number_of_patches=5):
    for i in range(0, number_of_patches):
        f, axarr = plt.subplots(1, 2)

        rand_index = random.randrange(0, len(data) - 1)
        data_example = data[rand_index]
        label_example = label[rand_index]
        axarr[0].imshow(data_example.reshape(data_example.shape[1], data_example.shape[2], data_example.shape[0]))
        axarr[1].imshow(label_example.reshape(label_example.shape[1], label_example.shape[2], label_example.shape[0]))
        plt.show()


# loads original images and creates smaller (default=33x33) patches from them
if __name__ == '__main__':
    label_patches = []
    patch_size = 2000
    # dataset can be found on https://drive.google.com/drive/folders/1B3DJGQKB6eNdwuQIhdskA64qUuVKLZ9u
    path = '../data/inputs/tutorial/T91/'
    dir_contents = listdir(path)
    print(f'Number of files: {len(dir_contents)}')
    patches_total = patch_size * len(dir_contents)
    for image_file in tqdm(dir_contents):
        filepath = path + image_file
        image_file = image.imread(filepath)
        # print(f'Image ({filepath} shape: {np.shape(image_file)}')
        label_patches.extend(create_patches(image_file, create_random_sample=True, max_size=patch_size))

    if len(label_patches) != patches_total:
        print(f'Total number of patches={len(label_patches)} is not equal to '
              f'patch_size={patch_size} * number of images={len(dir_contents)}')
    print(f'Total patches size: {np.shape(label_patches)}')

    # for i in range(0, 10):
    #     patch = label_patches[random.randrange(0, len(label_patches) - 1)]
    #     plt.imshow(patch)
    #     plt.show()

    save_patches_to_HDF5('../data/inputs/tutorial/train_mscale_3ch.h5', [], label_patches)

    # read_test_file = h5py.File('../data/inputs/tutorial/train_mscale_3ch.h5', mode='r')
    # label = read_test_file['label'][:]
    # print(np.shape(label))

    sys.exit()
