import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


IMG_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


def find_images(*path_parts):
    image_files = []
    for extension in IMG_EXTENSIONS:
        image_files.extend(glob(os.path.join(*path_parts, extension)))
    return sorted(image_files)


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if is_train:
            self.image_files = find_images(root, category, "train", "good")
        else:
            self.image_files = find_images(root, category, "test", "*")
            self.target_transform = transforms.Compose(
                [
                    transforms.Resize(
                        input_size, interpolation=transforms.InterpolationMode.NEAREST
                    ),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda target: (target > 0.5).float()),
                ]
            )
        self.is_train = is_train

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file).convert("RGB")
        image = self.image_transform(image)
        if self.is_train:
            return image
        else:
            if os.path.dirname(image_file).endswith("good"):
                target = torch.zeros([1, image.shape[-2], image.shape[-1]])
            else:
                mask_file = image_file.replace("/test/", "/ground_truth/")
                mask_file = os.path.splitext(mask_file)[0] + "_mask.png"
                target = Image.open(mask_file).convert("L")
                target = self.target_transform(target)
            return image, target

    def __len__(self):
        return len(self.image_files)
