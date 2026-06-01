import os
from glob import glob

import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


IMG_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")


class SquarePad:
    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image):
        width, height = image.size
        max_size = max(width, height)
        pad_left = (max_size - width) // 2
        pad_top = (max_size - height) // 2
        pad_right = max_size - width - pad_left
        pad_bottom = max_size - height - pad_top
        return transforms.functional.pad(
            image,
            padding=[pad_left, pad_top, pad_right, pad_bottom],
            fill=self.fill,
        )


def find_images(*path_parts):
    image_files = []
    for extension in IMG_EXTENSIONS:
        image_files.extend(glob(os.path.join(*path_parts, extension)))
    return sorted(image_files)


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, category, input_size, is_train=True):
        self.image_transform = transforms.Compose(
            [
                SquarePad(fill=0),
                transforms.Resize((input_size, input_size)),
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
                    SquarePad(fill=0),
                    transforms.Resize(
                        (input_size, input_size),
                        interpolation=transforms.InterpolationMode.NEAREST,
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
