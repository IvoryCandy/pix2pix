from os import listdir
from os.path import join

import torch.utils.data as data
import torchvision.transforms as transforms

import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(file_path):
    image = Image.open(file_path).convert('RGB')
    image = image.resize((512, 256), Image.BICUBIC)
    return image


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def get_training_set(root_dir):
    train_dir = join(root_dir, "train")
    return DatasetFromFolder(train_dir)


def get_test_set(root_dir):
    test_dir = join(root_dir, "test")
    return DatasetFromFolder(test_dir)


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir):
        super(DatasetFromFolder, self).__init__()
        self.image_path = image_dir
        self.image_filenames = [x for x in listdir(self.image_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        # Load Image
        img = load_img(join(self.image_path, self.image_filenames[index]))
        img = self.transform(img)

        w_total = img.size(2)
        w = int(w_total / 2)

        data = img[:, :, w:]
        target = img[:, :, :w]

        return data, target

    def __len__(self):
        return len(self.image_filenames)
