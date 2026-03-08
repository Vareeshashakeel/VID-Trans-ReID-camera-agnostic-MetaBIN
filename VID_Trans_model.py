import torch
import torch.nn as nn
import copy
from vit_ID import TransReID, Block
from functools import partial
from torch.nn import functional as F


def TCSS(features, shift, b, t):
    # features: [B*T, N, D]
    # reshape -> [B, N, T*D]
    features = features.reshape(b, features.size(1), t * features.size(2))
    token = features[:, 0:1]  # [B,1, t*D]

    # shift only patches (skip cls)
    patches = features[:, 1:]  # [B, N-1, t*D]
    patches = torch.cat([patches[:, shift:], patches[:, :shift]], dim=1)
    features = torch.cat([token, patches], dim=1)

    # Patch shuffling by 2-part (safe on odd length)
    patches = features[:, 1:]
    if patches.size(1) % 2 != 0:
        patches = torch.cat([patches, patches[:, -1:, :]], dim=1)

    batchsize = features.size(0)
    dim = features.size(-1)

    patches = patches.reshape(batchsize, 2, -1, dim)
    patches = torch.transpose(patches, 1, 2).contiguous()
    patches = patches.reshape(batchsize, -1, dim)

    return patches, token


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if getattr(m, "affine", False):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class MetaBIN1d(nn.Module):
    """Meta Batch-Instance Normalization for 1D feature vectors.
    Input/Output: [B, D]
    """

    def __init__(self, dim: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.bn = nn.BatchNorm1d(dim, affine=True)

        # gate per dim
        self.alpha = nn.Parameter(torch.zeros(dim))

        if affine:
            self.gamma = nn.Parameter(torch.ones(dim))
            self.beta = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    @staticmethod
    def _instance_norm_vec(x: torch.Tensor, eps: float) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise RuntimeError(f"MetaBIN1d expects 2D [B,D] input, got {tuple(x.shape)}")

        x_bn = self.bn(x)
        x_in = self._instance_norm_vec(x, self.eps)
        gate = torch.sigmoid(self.alpha).unsqueeze(0)  # [1,D]

        y = gate * x_bn + (1.0 - gate) * x_in
        if self.gamma is not None:
            y = y * self.gamma.unsqueeze(0) + self.beta.unsqueeze(0)
        return y


class VID_Trans(nn.Module):
    def __init__(self, num_classes, camera_num=None, pretrainpath=None):
        super().__init__()
        self.in_planes = 768
        self.num_classes = num_classes

        # backbone
        self.base = TransReID(
            img_size=[256, 128],
            patch_size=16,
            stride_size=[16, 16],
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.1,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        if pretrainpath is not None:
            state_dict = torch.load(pretrainpath, map_location="cpu")
            self.base.load_param(state_dict, load=True)

        # global stream: last block + norm
        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        # MetaBIN necks
        self.bottleneck = MetaBIN1d(self.in_planes)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        # local stream
        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.block1 = Block(
            dim=3072,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop=0,
            attn_drop=0,
            drop_path=dpr[11],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        self.b2 = nn.Sequential(self.block1, nn.LayerNorm(3072))

        self.bottleneck_1 = MetaBIN1d(3072); self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = MetaBIN1d(3072); self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = MetaBIN1d(3072); self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = MetaBIN1d(3072); self.bottleneck_4.apply(weights_init_kaiming)

        self.classifier_1 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_4.apply(weights_init_classifier)

        # video attention
        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, kernel_size=1, stride=1)
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, kernel_size=3, padding=1)
        self.attention_conv.apply(weights_init_kaiming)
        self.attention_tconv.apply(weights_init_kaiming)

        self.shift_num = 5
        self.part = 4

        # stability knobs
        self.attn_logit_clip = 10.0
        self.attn_val_clip = 10.0

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        Train input: x [B,T,C,H,W]
        Test input: x [B,T,C,H,W]  (B could be num_clips in your test script)
        """
        if x.dim() != 5:
            raise RuntimeError(f"Expected 5D input [B,T,C,H,W], got {x.dim()}D: {tuple(x.shape)}")

        b = x.size(0)
        t = x.size(1)

        # flatten frames -> [B*T,C,H,W]
        x = x.reshape(b * t, x.size(2), x.size(3), x.size(4))

        # backbone -> [B*T, N, 768]
        features = self.base(x)

        # -------------------------
        # Global branch
        # -------------------------
        b1_feat = self.b1(features)
        global_token = b1_feat[:, 0]  # [B*T,768]

        # attention over time
        g = global_token.unsqueeze(-1).unsqueeze(-1)  # [B*T,768,1,1]
        a = F.relu(self.attention_conv(g))            # [B*T,256,1,1]
        a = a.reshape(b, t, self.middle_dim).permute(0, 2, 1).contiguous()  # [B,256,T]
        a = F.relu(self.attention_tconv(a)).squeeze(1)  # [B,T]

        a_vals = torch.clamp(a, 0.0, self.attn_val_clip)
        a_logits = torch.clamp(a, 0.0, self.attn_logit_clip)
        a_w = F.softmax(a_logits, dim=1)  # [B,T]

        frame_feats = global_token.reshape(b, t, self.in_planes)  # [B,T,768]
        global_feat = (frame_feats * a_w.unsqueeze(-1)).sum(dim=1)  # [B,768]

        feat_bn = self.bottleneck(global_feat)  # for classification

        # -------------------------
        # Local parts (patch stream)
        # -------------------------
        feature_length = features.size(1) - 1
        patch_length = feature_length // 4

        patches, token = TCSS(features, self.shift_num, b, t)

        p1 = patches[:, :patch_length]
        p2 = patches[:, patch_length:patch_length * 2]
        p3 = patches[:, patch_length * 2:patch_length * 3]
        p4 = patches[:, patch_length * 3:patch_length * 4]

        part1 = self.b2(torch.cat((token, p1), dim=1)); part1_f = part1[:, 0]
        part2 = self.b2(torch.cat((token, p2), dim=1)); part2_f = part2[:, 0]
        part3 = self.b2(torch.cat((token, p3), dim=1)); part3_f = part3[:, 0]
        part4 = self.b2(torch.cat((token, p4), dim=1)); part4_f = part4[:, 0]

        part1_bn = self.bottleneck_1(part1_f)
        part2_bn = self.bottleneck_2(part2_f)
        part3_bn = self.bottleneck_3(part3_f)
        part4_bn = self.bottleneck_4(part4_f)

        if self.training:
            # classification uses BN/MetaBIN features
            Global_ID = self.classifier(feat_bn)
            Local_ID1 = self.classifier_1(part1_bn)
            Local_ID2 = self.classifier_2(part2_bn)
            Local_ID3 = self.classifier_3(part3_bn)
            Local_ID4 = self.classifier_4(part4_bn)

            # metric losses should see raw features (more standard)
            feat_list = [
                F.normalize(global_feat, p=2, dim=1),
                F.normalize(part1_f, p=2, dim=1),
                F.normalize(part2_f, p=2, dim=1),
                F.normalize(part3_f, p=2, dim=1),
                F.normalize(part4_f, p=2, dim=1),
            ]
            return [Global_ID, Local_ID1, Local_ID2, Local_ID3, Local_ID4], feat_list, a_vals

        else:
            # test embedding: concat BN features + normalize (important!)
            emb = torch.cat([feat_bn, part1_bn / 4, part2_bn / 4, part3_bn / 4, part4_bn / 4], dim=1)
            emb = F.normalize(emb, p=2, dim=1)
            return emb

    def load_param(self, trained_path, load=False):
        if not load:
            param_dict = torch.load(trained_path)
            for i in param_dict:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
            print(f'Loading pretrained model from {trained_path}')
        else:
            param_dict = trained_path
            for i in param_dict:
                if i not in self.state_dict() or 'classifier' in i or 'sie_embed' in i:
                    continue
                self.state_dict()[i].copy_(param_dict[i])
