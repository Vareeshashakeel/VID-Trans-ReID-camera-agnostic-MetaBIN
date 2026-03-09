import torch
import torch.nn.functional as F
from loss.softmax_loss import CrossEntropyLabelSmooth
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss


def make_loss(num_classes: int):
    """
    ID loss + Triplet loss + Center loss (camera-agnostic).

    IMPORTANT:
    - Triplet should use normalized embeddings.
    - Center loss should use RAW embeddings (NOT normalized).
    """

    feat_dim = 768
    feat_dim_aux = 3072  # keep if your CenterLoss expects it; otherwise not used

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
    # NOTE: Your part features are 3072-d, so if you want center on parts, use feat_dim_aux
    center_criterion_aux = CenterLoss(num_classes=num_classes, feat_dim=feat_dim_aux)

    triplet = TripletLoss()
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)

    def loss_func(score, feat, target):
        # -------------------------
        # ID (softmax) loss
        # -------------------------
        if isinstance(score, list):
            id_loss_main = xent(score[0], target)
            if len(score) > 1:
                id_loss_aux = sum(xent(s, target) for s in score[1:]) / len(score[1:])
                id_loss = 0.75 * id_loss_main + 0.25 * id_loss_aux
            else:
                id_loss = id_loss_main
        else:
            id_loss = xent(score, target)

        # -------------------------
        # Triplet + Center loss
        # -------------------------
        if isinstance(feat, list):
            # feat are RAW from model (after our fix)

            # Triplet on normalized
            f0 = F.normalize(feat[0], p=2, dim=1)
            tri_main = triplet(f0, target)[0]

            if len(feat) > 1:
                tri_aux = 0.0
                for f in feat[1:]:
                    tri_aux += triplet(F.normalize(f, p=2, dim=1), target)[0]
                tri_aux = tri_aux / len(feat[1:])
                tri_loss = 0.75 * tri_main + 0.25 * tri_aux
            else:
                tri_loss = tri_main

            # Center on RAW (do NOT normalize)
            center_main = center_criterion(feat[0], target)

            if len(feat) > 1:
                # parts are 3072-d so use aux center criterion
                center_aux = 0.0
                for f in feat[1:]:
                    center_aux += center_criterion_aux(f, target)
                center_aux = center_aux / len(feat[1:])
                center_loss = 0.75 * center_main + 0.25 * center_aux
            else:
                center_loss = center_main

        else:
            # single feature case
            tri_loss = triplet(F.normalize(feat, p=2, dim=1), target)[0]
            center_loss = center_criterion(feat, target)

        return id_loss + tri_loss, center_loss

    return loss_func, center_criterion
