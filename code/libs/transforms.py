import random
import math

import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms import functional as F
import torchvision.transforms as T


class Compose:
    """
    Compose a set of transforms that are jointly applied to
    input image and its corresponding detection annotations (e.g., boxes)
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ConvertAnnotations:
    """
    Convert the COCO annotations into a format that can be accepted by the model.
    The converted target include
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
            ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W``
            and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - image_id (Tensor[1]): the id of the image
        - area (Tensor[N]): area of each ground-truth box (not used)
        - iscrowd (Tensor[N]): if the box contains multiple objects (not used)
    """

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64) - 1

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Random horizontal flip of the image and boxes.
    """

    def forward(self, image, target):
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            if target is not None:
                _, _, width = F.get_dimensions(image)
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target


class RandomColorJitter(T.ColorJitter):
    """
    Randomly change the brightness, contrast, saturation and hue of an image.
    """

    def forward(self, image, target):
        image = super().forward(image)
        return image, target
    

class RandomResizedCrop(T.RandomResizedCrop):
    """
    Random resized crop of the image and boxes.
    
    This class mimics the interface of RandomHorizontalFlip by accepting
    both image and target, and correctly transforming the bounding boxes.
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=F.InterpolationMode.BILINEAR):
        """
        Args:
            size (int or sequence): desired output size.
            scale (tuple of float): range of size of the origin size cropped
            ratio (tuple of float): range of aspect ratio of the origin aspect ratio cropped
            interpolation (InterpolationMode): Desired interpolation.
        """
        super().__init__(size=size, scale=scale, ratio=ratio, interpolation=interpolation)

    def forward(self, image, target):
        """
        Args:
            image (PIL Image or Tensor): Image to be cropped and resized.
            target (dict): The target annotations.

        Returns:
            Tuple: tuple (image, target)
        """
        
        i, j, h, w = self.get_params(image, self.scale, self.ratio)

        transformed_image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)

        if target is not None:
            boxes = target["boxes"] # [N, 4] (x1, y1, x2, y2)
            
            if boxes.shape[0] == 0:
                return transformed_image, target

            crop_x1 = j
            crop_y1 = i
            crop_x2 = j + w
            crop_y2 = i + h
            crop_window = torch.tensor([crop_x1, crop_y1, crop_x2, crop_y2], device=boxes.device, dtype=boxes.dtype)

            inter_x1 = torch.max(boxes[:, 0], crop_window[0])
            inter_y1 = torch.max(boxes[:, 1], crop_window[1])
            inter_x2 = torch.min(boxes[:, 2], crop_window[2])
            inter_y2 = torch.min(boxes[:, 3], crop_window[3])

            inter_w = inter_x2 - inter_x1
            inter_h = inter_y2 - inter_y1
            keep = (inter_w > 0) & (inter_h > 0)
            
            if not keep.any():
                target["boxes"] = torch.empty((0, 4), device=boxes.device, dtype=boxes.dtype)
                target["labels"] = torch.empty((0,), device=target["labels"].device, dtype=target["labels"].dtype)
                return transformed_image, target

            boxes = boxes[keep]
            target["labels"] = target["labels"][keep]
            
            clipped_boxes = torch.stack([
                inter_x1[keep], inter_y1[keep], inter_x2[keep], inter_y2[keep]
            ], dim=1)

            clipped_boxes[:, 0::2] = clipped_boxes[:, 0::2] - crop_x1
            clipped_boxes[:, 1::2] = clipped_boxes[:, 1::2] - crop_y1

            if isinstance(self.size, int):
                new_h = self.size
                new_w = self.size
            else:
                new_h, new_w = self.size

            scale_x = new_w / w
            scale_y = new_h / h

            clipped_boxes[:, 0] = clipped_boxes[:, 0] * scale_x
            clipped_boxes[:, 1] = clipped_boxes[:, 1] * scale_y
            clipped_boxes[:, 2] = clipped_boxes[:, 2] * scale_x
            clipped_boxes[:, 3] = clipped_boxes[:, 3] * scale_y
            
            scaled_w = clipped_boxes[:, 2] - clipped_boxes[:, 0]
            scaled_h = clipped_boxes[:, 3] - clipped_boxes[:, 1]
            min_size = 1.0 
            keep_final = (scaled_w > min_size) & (scaled_h > min_size)

            target["boxes"] = clipped_boxes[keep_final]
            target["labels"] = target["labels"][keep_final]

        return transformed_image, target


class ToTensor(nn.Module):
    """
    Convert an image (PIL or np.array) to tensor.
    This function will additional perform normalization so that each pixel value
    is a floating point number in the range of [0, 1].
    """

    def forward(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    Args:
        tensors (tensor): Tensor containing images.
        image_sizes (list[tuple[int, int]]): List of Tuples each containing size of images.
    """

    def __init__(self, tensors, image_sizes):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        return ImageList(cast_tensor, self.image_sizes)


def _resize_image(image, img_min_size, img_max_size):
    """
    Resize an image such that its shortest side = img_min_size
    and its largest side is <= img_max_size
    """
    im_shape = torch.tensor(image.shape[-2:])
    min_size = torch.min(im_shape).to(dtype=torch.float32)
    max_size = torch.max(im_shape).to(dtype=torch.float32)
    scale = torch.min(img_min_size / min_size, img_max_size / max_size)
    scale_factor = scale.item()

    image = torch.nn.functional.interpolate(
        image[None],
        size=None,
        scale_factor=scale_factor,
        mode="bilinear",
        recompute_scale_factor=True,
        align_corners=False,
    )[0]

    return image


def _resize_boxes(boxes, original_size, new_size):
    """
    Resize a set of boxes based on the scaling factors
    of their corresponding images
    """
    ratios = [
        torch.tensor(s, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)

    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)


class GeneralizedRCNNTransform(nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedRCNN
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets

    Args:
        img_min_size (List[int]): a set of minimum size of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.
        size_divisible (int): each size of an input image must be divisible by
        this number, otherwise padding is needed.
    """

    def __init__(
        self,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        size_divisible=32,
    ):
        super().__init__()
        if not isinstance(img_min_size, (list, tuple)):
            img_min_size = (img_min_size,)
        self.min_size = img_min_size
        self.max_size = img_max_size
        self.image_mean = img_mean
        self.image_std = img_std
        self.size_divisible = size_divisible

    def forward(self, images, targets):
        images = [img for img in images]

        for i in range(len(images)):
            image = images[i]
            target_index = targets[i] if targets is not None else None

            if image.dim() != 3:
                raise ValueError(
                    f"images is expected to be a list of 3d tensors of shape [C, H, W], "
                    f"got {image.shape}"
                )
            image = self.normalize(image)
            image, target_index = self.resize(image, target_index)
            images[i] = image
            if targets is not None and target_index is not None:
                targets[i] = target_index

        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_sizes_list = []
        for image_size in image_sizes:
            torch._assert(
                len(image_size) == 2,
                f"Input tensors expected to have in the last two elements H and W, "
                f"instead got {image_size}",
            )
            image_sizes_list.append((image_size[0], image_size[1]))

        image_list = ImageList(images, image_sizes_list)
        return image_list, targets

    def normalize(self, image):
        if not image.is_floating_point():
            raise TypeError(
                f"Expected input images to be of floating type (in range [0, 1]), "
                f"but found type {image.dtype} instead"
            )
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        if self.training:
            size = float(random.choice(self.min_size))
        else:
            # assume for now that testing uses the largest scale
            size = float(self.min_size[-1])
        image = _resize_image(image, size, float(self.max_size))

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = _resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox
        # also update the area to avoid confusion
        target["area"] = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])

        return image, target

    def max_by_axis(self, the_list):
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def batch_images(self, images):
        max_size = self.max_by_axis([list(img.shape) for img in images])
        stride = float(self.size_divisible)
        max_size = list(max_size)
        max_size[1] = int(math.ceil(float(max_size[1]) / stride) * stride)
        max_size[2] = int(math.ceil(float(max_size[2]) / stride) * stride)

        batch_shape = [len(images)] + max_size
        batched_imgs = images[0].new_full(batch_shape, 0)
        for i in range(batched_imgs.shape[0]):
            img = images[i]
            batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(
            zip(result, image_shapes, original_image_sizes)
        ):
            boxes = pred["boxes"]
            boxes = _resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes
        return result

    def __repr__(self):
        format_string = f"{self.__class__.__name__}("
        _indent = "\n    "
        format_string += (
            f"{_indent}Normalize(mean={self.image_mean}, std={self.image_std})"
        )
        format_string += f"{_indent}Resize(min_size={self.min_size}, max_size={self.max_size}, mode='bilinear')"
        format_string += "\n)"
        return format_string
