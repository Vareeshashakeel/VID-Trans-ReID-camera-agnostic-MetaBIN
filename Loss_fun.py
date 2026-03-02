import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss(num_classes: int):
    """
    Build ID loss + Triplet loss (+ Center loss).

    Camera labels are NOT used.
    This loss is fully camera-agnostic.
    """

    feat_dim = 768      # CLS token feature dim
    feat_dim_aux = 3072 # multi-part / concatenated feature dim (if used)

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
    center_criterion_aux = CenterLoss(num_classes=num_classes, feat_dim=feat_dim_aux)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):
        """
        Args:
            score: logits (Tensor or list of Tensors)
            feat: features (Tensor or list of Tensors)
            target: ground-truth IDs
        """

        # -------------------------
        # ID (softmax) loss
        # -------------------------
        if isinstance(score, list):
            id_loss = xent(score[0], target)
            if len(score) > 1:
                aux_loss = sum(xent(s, target) for s in score[1:]) / len(score[1:])
                id_loss = 0.75 * id_loss + 0.25 * aux_loss
        else:
            id_loss = xent(score, target)

        # -------------------------
        # Triplet + Center loss
        # -------------------------
        if isinstance(feat, list):
            tri_loss = triplet(feat[0], target)[0]
            if len(feat) > 1:
                aux_tri = sum(triplet(f, target)[0] for f in feat[1:]) / len(feat[1:])
                tri_loss = 0.75 * tri_loss + 0.25 * aux_tri

            center_loss = center_criterion(feat[0], target)
            if len(feat) > 1:
                aux_center = sum(center_criterion_aux(f, target) for f in feat[1:]) / len(feat[1:])
                center_loss = 0.75 * center_loss + 0.25 * aux_center
        else:
            tri_loss = triplet(feat, target)[0]
            center_loss = 0.0

        return id_loss + tri_loss, center_loss

    return loss_func, center_criterion
