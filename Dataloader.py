import os
import random
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from utility import RandomIdentitySampler, RandomErasing3
from Datasets.MARS_dataset import Mars
from Datasets.iLDSVID import iLIDSVID
from Datasets.PRID_dataset import PRID

__factory = {
    'Mars': Mars,
    'iLIDSVID': iLIDSVID,
    'PRID': PRID,
}

# ---------------------------------------------------------
# Collate functions
# ---------------------------------------------------------

def train_collate_fn(batch):
    """
    Returns:
        imgs: [B, T, C, H, W]
        pids: [B]
        camids: [B]   (returned for protocol only; model may ignore)
        erase_mask: [B, T]
    """
    imgs, pids, camids, erase_mask = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        torch.tensor(pids, dtype=torch.long),
        torch.tensor(camids, dtype=torch.long),
        torch.stack(erase_mask, dim=0),
    )


def val_collate_fn(batch):
    """
    Returns:
        imgs: [B, K, T, C, H, W]  (B=1 during eval)
        pids: list[int]
        camids: [B]
        img_paths: list
    """
    imgs, pids, camids, img_paths = zip(*batch)
    return (
        torch.stack(imgs, dim=0),
        list(pids),
        torch.tensor(camids, dtype=torch.long),
        img_paths,
    )


# ---------------------------------------------------------
# Public dataloader function
# ---------------------------------------------------------

def dataloader(
    Dataset_name,
    *,
    batch_size=64,
    seq_len=4,
    num_workers=4,
    pin_memory=True,
):
    """
    Camera-agnostic dataloader.
    camids are returned for evaluation protocol (filter same id+cam).
    """

    train_transforms = T.Compose([
        T.Resize((256, 128)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    val_transforms = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    dataset = __factory[Dataset_name]()

    train_set = VideoDatasetTrain(
        dataset.train,
        seq_len=seq_len,
        transform=train_transforms,
        sample="intelligent",     # ✅ matches base behavior
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomIdentitySampler(dataset.train, batch_size, seq_len),
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=train_collate_fn,
        drop_last=True,
    )

    query_set = VideoDatasetTest(
        dataset.query,
        seq_len=seq_len,
        transform=val_transforms,
    )

    gallery_set = VideoDatasetTest(
        dataset.gallery,
        seq_len=seq_len,
        transform=val_transforms,
    )

    query_loader = DataLoader(
        query_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate_fn,
    )

    gallery_loader = DataLoader(
        gallery_set,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=val_collate_fn,
    )

    return (
        train_loader,
        len(dataset.query),
        dataset.num_train_pids,
        dataset.num_train_cams,
        dataset.num_train_vids,
        query_loader,
        gallery_loader,
    )


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def read_image(img_path):
    while True:
        try:
            return Image.open(img_path).convert('RGB')
        except IOError:
            print(f"Retry reading image: {img_path}")


# ---------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------

class VideoDatasetTrain(Dataset):
    """
    Training dataset with random erasing + "intelligent" sampling
    """

    def __init__(self, dataset, seq_len, transform=None, sample="intelligent"):
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform
        self.sample = sample
        self.erase = RandomErasing3(probability=0.5)

    def __len__(self):
        return len(self.dataset)

    def _intelligent_indices(self, num):
        """
        Split tracklet into seq_len segments; pick 1 random frame from each segment.
        This matches the base repo behavior and increases temporal diversity.
        """
        indices = []
        each = max(num // self.seq_len, 1)
        for i in range(self.seq_len):
            if i != self.seq_len - 1:
                start = min(i * each, num - 1)
                end = min((i + 1) * each - 1, num - 1)
                indices.append(random.randint(start, end))
            else:
                start = min(i * each, num - 1)
                indices.append(random.randint(start, num - 1))
        return np.array(indices, dtype=int)

    def __getitem__(self, idx):
        img_paths, pid, camid = self.dataset[idx]
        num = len(img_paths)

        if self.sample == "intelligent":
            indices = self._intelligent_indices(num)
        else:
            # fallback: evenly spaced (NOT recommended for training)
            indices = np.linspace(0, num - 1, self.seq_len).astype(int)

        imgs = []
        erase_mask = []

        for i in indices:
            img = read_image(img_paths[i])
            if self.transform:
                img = self.transform(img)
            img, mask = self.erase(img)
            imgs.append(img.unsqueeze(0))
            erase_mask.append(mask)

        imgs = torch.cat(imgs, dim=0)
        erase_mask = torch.tensor(erase_mask, dtype=torch.float32)

        return imgs, pid, camid, erase_mask


class VideoDatasetTest(Dataset):
    """
    Dense sampling for evaluation
    """

    def __init__(self, dataset, seq_len, transform=None, max_length=40):
        self.dataset = dataset
        self.seq_len = seq_len
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_paths, pid, camid = self.dataset[idx]
        num = len(img_paths)

        clips = []
        cur = 0

        while cur < num:
            end = min(cur + self.seq_len, num)
            inds = list(range(cur, end))
            if len(inds) < self.seq_len:
                inds += [inds[-1]] * (self.seq_len - len(inds))

            imgs = []
            for i in inds:
                img = read_image(img_paths[i])
                if self.transform:
                    img = self.transform(img)
                imgs.append(img.unsqueeze(0))

            clips.append(torch.cat(imgs, dim=0))
            cur += self.seq_len
            if len(clips) >= self.max_length:
                break

        clips = torch.stack(clips)  # [K, T, C, H, W]
        return clips, pid, camid, img_paths
