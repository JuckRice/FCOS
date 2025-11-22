import os
import random

import numpy as np

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .transforms import Compose, ConvertAnnotations, RandomHorizontalFlip, RandomColorJitter, RandomResizedCrop, ToTensor


def trivial_batch_collator(batch):
    """
    A batch collator that allows us to bypass auto batching
    """
    return tuple(zip(*batch))


def worker_init_reset_seed(worker_id):
    """
    Reset random seed for each worker
    """
    seed = torch.initial_seed() % 2**31
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class VOCDetection(torchvision.datasets.CocoDetection):
    """
    A simple dataset wrapper to load VOC data
    """

    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def get_cls_names(self):
        cls_names = (
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        )
        return cls_names

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


class COCODetection(torchvision.datasets.CocoDetection):
    """
    A wrapper for COCO dataset to format the target specific to our ConvertAnnotations transform.
    """
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        coco_ids = sorted(self.coco.getCatIds())
        self.id_map = {coco_id: i + 1 for i, coco_id in enumerate(coco_ids)}

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        valid_annotations = []
        for obj in target:
            original_id = obj["category_id"]
            if original_id in self.id_map:
                    obj["category_id"] = self.id_map[original_id]
                    valid_annotations.append(obj)
            else:
                pass
        
        target = dict(image_id=image_id, annotations=valid_annotations)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        return img, target


def build_dataset(name, split, img_folder, json_folder):
    """
    Create VOC dataset with default transforms for training / inference.
    New datasets can be linked here.
    """
    if name == "VOC2007":
        assert split in ["trainval", "test"]
        is_training = split == "trainval"
    elif name == "COCO":
        is_training = ("train" in split)
        ann_file = os.path.join(json_folder, f"instances_{split}.json")
        root = os.path.join(img_folder, split)
    else:
        print("Unsupported dataset")
        return None

    if is_training:
        transforms = Compose([ConvertAnnotations(), 
                              RandomColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                              RandomHorizontalFlip(), 
                              RandomResizedCrop(size=(800, 800), scale=(0.1, 1.0)),
                              ToTensor()])
    else:
        transforms = Compose([ConvertAnnotations(), ToTensor()])

    if name == "VOC2007":
        dataset = VOCDetection(
            img_folder, os.path.join(json_folder, split + ".json"), transforms
        )
    elif name == "COCO":
        dataset = COCODetection(
            img_folder=root, ann_file=ann_file, transforms=transforms
        )
    return dataset


def build_dataloader(dataset, is_training, batch_size, num_workers):
    """
    Create a dataloder for VOC dataset
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator,
        worker_init_fn=(worker_init_reset_seed if is_training else None),
        shuffle=is_training,
        drop_last=is_training,
        persistent_workers=True,
    )
    return loader
