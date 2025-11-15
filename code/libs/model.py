import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn
import torch.nn.functional as F

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the featuer map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        all_cls_logits = []
    
        for features in x:
            # apply the conv layers
            cls_features = self.conv(features)
            # apply the final classification layer
            cls_logits = self.cls_logits(cls_features)
            # append to the output list
            all_cls_logits.append(cls_logits)
            
        # [N, num_classes, H, W]
        return all_cls_logits


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU())
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        all_bbox_reg = []
        all_bbox_ctrness = []
        for features in x:
            # apply the conv layers
            reg_features = self.conv(features)
            # apply the bbox regression layer
            bbox_reg = self.bbox_reg(reg_features)
            # apply the center-ness layer               
            bbox_ctrness = self.bbox_ctrness(reg_features)
            
            all_bbox_reg.append(bbox_reg)
            all_bbox_ctrness.append(bbox_ctrness)

        # all_bbox_reg: list，shape of tensor [N, 4, H, W] (l, t, r, b)
        # all_bbox_ctrness: list，shape of tensor [N, 1, H, W]
        return all_bbox_reg, all_bbox_ctrness


class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function depends on if the model is in training
    or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detection results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)
    * You might want to double check the format of 2D coordinates saved in points

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        N = cls_logits[0].shape[0] # batch size
        num_classes = self.num_classes

        # 1. prepare/flatten predictions and points from all FPN levels
        # concatenate predictions and points across levels for batched processing
        all_cls_logits = []
        all_reg_outputs = []
        all_ctr_logits = []
        all_points = []
        all_strides = []
        all_reg_ranges = []

        for i, (cls_l, reg_o, ctr_l) in enumerate(zip(cls_logits, reg_outputs, ctr_logits)):
            # cls_l: [N, C, H, W], reg_o: [N, 4, H, W], ctr_l: [N, 1, H, W]
            H, W = cls_l.shape[-2:]
            
            # Rearrange and flatten: [N, C, H, W] -> [N, H, W, C] -> [N, H*W, C]
            all_cls_logits.append(cls_l.permute(0, 2, 3, 1).reshape(N, -1, num_classes))
            all_reg_outputs.append(reg_o.permute(0, 2, 3, 1).reshape(N, -1, 4))
            all_ctr_logits.append(ctr_l.permute(0, 2, 3, 1).reshape(N, -1, 1))
            
            # Flatten points, strides, and regression ranges
            # points[i] is [H, W, 2]
            level_points = points[i][:H, :W].reshape(-1, 2) # [H*W, 2]
            level_strides = strides[i].repeat(H * W) # [H*W]
            level_reg_range = reg_range[i].repeat(H * W, 1) # [H*W, 2]
            
            all_points.append(level_points)
            all_strides.append(level_strides)
            all_reg_ranges.append(level_reg_range)

        # Concatenate across levels
        cat_cls_logits = torch.cat(all_cls_logits, dim=1) # [N, sum(H*W), C]
        cat_reg_outputs = torch.cat(all_reg_outputs, dim=1) # [N, sum(H*W), 4]
        cat_ctr_logits = torch.cat(all_ctr_logits, dim=1) # [N, sum(H*W), 1]
        
        cat_points = torch.cat(all_points, dim=0) # [sum(H*W), 2]
        cat_strides = torch.cat(all_strides, dim=0) # [sum(H*W)]
        cat_reg_ranges = torch.cat(all_reg_ranges, dim=0) # [sum(H*W), 2]

        # Extract (x, y) coordinates from 'points'
        # As point_generator.py defines: (height, width) -> (y, x)
        points_x = cat_points[:, 1] # [P]
        points_y = cat_points[:, 0] # [P]
        
        # [P, 2] -> [P]
        reg_range_min = cat_reg_ranges[:, 0]
        reg_range_max = cat_reg_ranges[:, 1]
        
        P = cat_points.shape[0] # P = sum(H*W), total number of points

        # 2. Assign GT targets to each point
        # This must be done per image in the batch
        all_cls_targets = []
        all_reg_targets = []
        all_ctr_targets = []
        
        num_pos = 0.0 # total number of positive samples used for normalization

        for i in range(N): # iterate over each image in the batch
            gt_boxes = targets[i]['boxes'] # [M, 4] (x1, y1, x2, y2)
            gt_labels = targets[i]['labels'] # [M] (0-indexed)
            
            M = gt_boxes.shape[0] # number of GT boxes

            # Initialize targets for this image
            cls_target_img = cat_points.new_zeros((P, num_classes))
            reg_target_img = cat_points.new_zeros((P, 4))
            ctr_target_img = cat_points.new_zeros((P, 1))
            
            if M == 0:
                # No GT boxes: all points are background
                all_cls_targets.append(cls_target_img)
                all_reg_targets.append(reg_target_img)
                all_ctr_targets.append(ctr_target_img)
                continue

            # --- FCOS target assignment start ---
            
            # (1) Compute (l, t, r, b) distances from all points to all GT boxes
            # [P] -> [P, 1]; [M] -> [1, M] => [P, M]
            l = points_x[:, None] - gt_boxes[:, 0][None, :]
            t = points_y[:, None] - gt_boxes[:, 1][None, :]
            r = gt_boxes[:, 2][None, :] - points_x[:, None]
            b = gt_boxes[:, 3][None, :] - points_y[:, None]
            
            ltrb_targets = torch.stack([l, t, r, b], dim=2) # [P, M, 4]
            
            # (2) Rule 1: is the point inside the GT box?
            is_in_box = ltrb_targets.min(dim=2).values > 0 # [P, M]
            
            # (3) Rule 2: is the point within the regression range?
            max_ltrb = ltrb_targets.max(dim=2).values # [P, M]
            is_in_range = (max_ltrb >= reg_range_min[:, None]) & \
                          (max_ltrb <= reg_range_max[:, None]) # [P, M]
                          
            # (4) Rule 3: is the point within the center sampling region?
            gt_boxes_cx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
            gt_boxes_cy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
            
            dist_x = (points_x[:, None] - gt_boxes_cx[None, :]).abs() # [P, M]
            dist_y = (points_y[:, None] - gt_boxes_cy[None, :]).abs() # [P, M]
            
            center_radius = self.center_sampling_radius * cat_strides[:, None] # [P, 1]
            is_in_center = (dist_x < center_radius) & (dist_y < center_radius) # [P, M]
            
            # --- Combine all conditions ---
            is_positive_potential = is_in_box & is_in_range & is_in_center # [P, M]
            
            # (5) Rule 4: resolve ambiguity (assign to GT with smallest area)
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]) # [M]
            candidate_areas = gt_areas[None, :].repeat(P, 1) # [P, M]
            # Set non-positive candidates' area to infinity
            candidate_areas[~is_positive_potential] = float('inf')
            
            # Find the smallest-area GT for each point
            min_vals, min_indices = candidate_areas.min(dim=1) # [P], [P]
            
            # --- Finalize targets ---
            pos_mask = min_vals < float('inf') # [P], final positive mask
            num_pos_img = pos_mask.sum().item()
            num_pos += num_pos_img

            if num_pos_img > 0:
                # GT indices assigned to positive points
                assigned_gt_indices = min_indices[pos_mask] # [num_pos_img]
                
                # Assign classification targets
                assigned_labels = gt_labels[assigned_gt_indices] # [num_pos_img]
                cls_target_img[pos_mask] = F.one_hot(assigned_labels, num_classes=num_classes).float()
                
                # Assign regression targets (l, t, r, b)
                assigned_ltrb = ltrb_targets[pos_mask, assigned_gt_indices] # [num_pos_img, 4]
                # Normalize by stride (as in Eq. 1)
                assigned_strides = cat_strides[pos_mask] # [num_pos_img]
                reg_target_img[pos_mask] = assigned_ltrb / assigned_strides[:, None]
                
                # Assign centerness targets (as in Eq. 3)
                l_star, t_star, r_star, b_star = assigned_ltrb.unbind(dim=1)
                ctr_num = (l_star.min(r_star)) * (t_star.min(b_star))
                ctr_den = (l_star.max(r_star)) * (t_star.max(b_star))
                ctr_target_val = torch.sqrt(ctr_num / (ctr_den + 1e-6)) # [num_pos_img]
                ctr_target_img[pos_mask] = ctr_target_val.unsqueeze(-1) # [num_pos_img, 1]

            # Append to batch lists
            all_cls_targets.append(cls_target_img)
            all_reg_targets.append(reg_target_img)
            all_ctr_targets.append(ctr_target_img)
        
        # 3. Compute losses

        # Stack targets from all images
        batch_cls_targets = torch.stack(all_cls_targets) # [N, P, C]
        batch_reg_targets = torch.stack(all_reg_targets) # [N, P, 4]
        batch_ctr_targets = torch.stack(all_ctr_targets) # [N, P, 1]

        # Normalize by number of positive samples (as in Eq. 2)
        # Ensure num_pos at least 1 to avoid division by zero
        num_pos = max(1.0, num_pos)
        
        # (1) Classification loss (Focal Loss)
        # computed over all points
        # [N, P, C] -> [N*P, C]
        cls_loss = sigmoid_focal_loss(
            cat_cls_logits.reshape(-1, num_classes),
            batch_cls_targets.reshape(-1, num_classes),
            reduction="sum"
        ) / num_pos
        
        # (2) Regression loss (GIoU Loss)
        # (3) Centerness loss (BCE Loss)
        # computed only for positive points

        # Get mask of positive samples in the batch [N, P]
        batch_pos_mask = batch_cls_targets.max(dim=2).values > 0
        
        # If no positive samples, losses are zero
        if num_pos > 0:
            # --- Regression loss ---
            # Select predictions and targets for positive samples [sum(num_pos_img), 4]
            pred_boxes_ltrb = cat_reg_outputs[batch_pos_mask]
            target_boxes_ltrb = batch_reg_targets[batch_pos_mask]
            
            # Need to convert l,t,r,b back to x1,y1,x2,y2 for giou_loss
            # [N, P] -> [sum(num_pos_img)]
            batch_points_x = points_x.expand(N, P)[batch_pos_mask]
            batch_points_y = points_y.expand(N, P)[batch_pos_mask]
            batch_strides = cat_strides.expand(N, P)[batch_pos_mask]
            
            # Un-normalize predicted values
            pred_ltrb_unnorm = pred_boxes_ltrb * batch_strides[:, None]
            pred_x1 = batch_points_x - pred_ltrb_unnorm[:, 0]
            pred_y1 = batch_points_y - pred_ltrb_unnorm[:, 1]
            pred_x2 = batch_points_x + pred_ltrb_unnorm[:, 2]
            pred_y2 = batch_points_y + pred_ltrb_unnorm[:, 3]
            pred_boxes_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
            
            # Un-normalize target values
            target_ltrb_unnorm = target_boxes_ltrb * batch_strides[:, None]
            target_x1 = batch_points_x - target_ltrb_unnorm[:, 0]
            target_y1 = batch_points_y - target_ltrb_unnorm[:, 1]
            target_x2 = batch_points_x + target_ltrb_unnorm[:, 2]
            target_y2 = batch_points_y + target_ltrb_unnorm[:, 3]
            target_boxes_xyxy = torch.stack([target_x1, target_y1, target_x2, target_y2], dim=1)
            
            reg_loss = giou_loss(
                pred_boxes_xyxy,
                target_boxes_xyxy,
                reduction="sum"
            ) / num_pos
            
            # --- Centerness loss ---
            # [sum(num_pos_img), 1]
            pred_ctr_logits = cat_ctr_logits[batch_pos_mask]
            target_ctr = batch_ctr_targets[batch_pos_mask]
            
            ctr_loss = F.binary_cross_entropy_with_logits(
                pred_ctr_logits,
                target_ctr,
                reduction="sum"
            ) / num_pos
            
        else:
            # No positive samples, losses are zero
            reg_loss = cat_reg_outputs.sum() * 0.0
            ctr_loss = cat_ctr_logits.sum() * 0.0

        # --- Final loss ---
        final_loss = cls_loss + reg_loss + ctr_loss
        
        return {
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "ctr_loss": ctr_loss,
            "final_loss": final_loss,
        }

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with low object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes and their labels
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep a fixed number of boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """

    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        detections = []
        N = cls_logits[0].shape[0] # batch size
        
        # iterate over each image in the batch
        for i in range(N):
            
            # padded image size (H_pad, W_pad)
            img_shape_padded = image_shapes[i] 
            
            boxes_per_image = []
            scores_per_image = []
            labels_per_image = []

            # (a) Loop over every pyramid level
            for level, (cls_l, reg_o, ctr_l) in enumerate(zip(cls_logits, reg_outputs, ctr_logits)):
                
                # 1. get the predictions for this image and level
                # [C, H, W], [4, H, W], [1, H, W]
                cls_l_img = cls_l[i]
                reg_o_img = reg_o[i]
                ctr_l_img = ctr_l[i]
                
                H, W = cls_l_img.shape[-2:]
                
                # get the points and stride for this level
                points_level = points[level][:H, :W] # [H, W, 2]
                stride_level = strides[level]

                # flacten the predictions and points
                # [C, H*W] -> [H*W, C]
                cls_l_flat = cls_l_img.reshape(self.num_classes, -1).permute(1, 0)
                # [4, H*W] -> [H*W, 4]
                reg_o_flat = reg_o_img.reshape(4, -1).permute(1, 0)
                # [1, H*W] -> [H*W, 1]
                ctr_l_flat = ctr_l_img.reshape(1, -1).permute(1, 0)
                # [H, W, 2] -> [H*W, 2] (y, x)
                points_flat = points_level.reshape(-1, 2)
                
                # (1) calculate the final object scores
                scores_cls = torch.sigmoid(cls_l_flat) # [H*W, C]
                scores_ctr = torch.sigmoid(ctr_l_flat) # [H*W, 1]
                # FCOS  sqrt(cls * ctr)
                scores_final = torch.sqrt(scores_cls * scores_ctr) # [H*W, C]
                
                # (2) & (3) Top-K
                # Don't reshape scores to [H*W, C] -> [H*W * C] yet, [H*W, C] 
                candidate_scores = scores_final.reshape(-1) # [H*W * C]
                
                # keep Top-K candidates before NMS
                num_topk = min(self.topk_candidates, candidate_scores.numel())
                topk_scores, topk_indices = torch.topk(candidate_scores, num_topk)
                
                # apply score threshold
                keep_mask = topk_scores > self.score_thresh
                topk_scores = topk_scores[keep_mask] # [num_kept]
                topk_indices = topk_indices[keep_mask] # [num_kept]
                
                if topk_scores.numel() == 0:
                    continue # no candidates left for this level

                # (4) decode boxes and labels
                # `topk_indices` is the index in [H*W * C] 
                # location index (0 ~ H*W-1)
                location_indices = topk_indices // self.num_classes
                # label index (0 ~ C-1)
                labels = topk_indices % self.num_classes
                
                # get the corresponding regression outputs and points
                reg_ltrb = reg_o_flat[location_indices] # [num_kept, 4]
                points_xy = points_flat[location_indices] # [num_kept, 2] (y, x)
                
                # output as l, t, r, b (unnormalized)
                reg_ltrb_unnorm = reg_ltrb * stride_level
                
                # decode as [x1, y1, x2, y2]
                # points_xy[:, 1] = x, points_xy[:, 0] = y
                pred_x1 = points_xy[:, 1] - reg_ltrb_unnorm[:, 0]
                pred_y1 = points_xy[:, 0] - reg_ltrb_unnorm[:, 1]
                pred_x2 = points_xy[:, 1] + reg_ltrb_unnorm[:, 2]
                pred_y2 = points_xy[:, 0] + reg_ltrb_unnorm[:, 3]
                
                boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
                scores = topk_scores
                
                # (5) clip boxes outside of the image boundaries / remove small boxes
                # `img_shape_padded` 是 (H, W)
                boxes[:, 0] = boxes[:, 0].clamp(min=0, max=img_shape_padded[1])
                boxes[:, 1] = boxes[:, 1].clamp(min=0, max=img_shape_padded[0])
                boxes[:, 2] = boxes[:, 2].clamp(min=0, max=img_shape_padded[1])
                boxes[:, 3] = boxes[:, 3].clamp(min=0, max=img_shape_padded[0])
                
                # remove samll boxes
                keep_mask_size = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
                boxes = boxes[keep_mask_size]
                scores = scores[keep_mask_size]
                labels = labels[keep_mask_size]
                
                # append to level lists
                boxes_per_image.append(boxes)
                scores_per_image.append(scores)
                labels_per_image.append(labels)
            
            # (b) collect all candidate boxes across all pyramid levels
            if not boxes_per_image:
                detections.append({
                    "boxes": torch.empty(0, 4, device=cls_l.device),
                    "scores": torch.empty(0, device=cls_l.device),
                    "labels": torch.empty(0, dtype=torch.long, device=cls_l.device)
                })
                continue

            final_boxes = torch.cat(boxes_per_image, dim=0)
            final_scores = torch.cat(scores_per_image, dim=0)
            final_labels = torch.cat(labels_per_image, dim=0)
            
            # (c) perform NMS
            keep_indices = batched_nms(final_boxes, final_scores, final_labels, self.nms_thresh)
            
            # (d) keep a fixed number of boxes after NMS
            keep_indices = keep_indices[:self.detections_per_img]
            
            final_boxes = final_boxes[keep_indices]
            final_scores = final_scores[keep_indices]
            final_labels = final_labels[keep_indices]
            
            # "Output labels should be offseted by +1"
            # compensate `dataset.py`  `ConvertAnnotations` which makes labels 0-indexed
            final_labels = final_labels + 1
            
            detections.append({
                "boxes": final_boxes,
                "scores": final_scores,
                "labels": final_labels
            })
            
        return detections
