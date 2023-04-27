import os#===========看如何写一份训练代码.

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from pycocotools.coco import COCO
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

#++++++++++++++建立数据集的类
class COCODataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.coco = COCO(annotation_file)#使用coco库包来读取标注文件.
        self.image_ids = list(self.coco.imgs.keys())

        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0] # ====只要那些带有anooation的图片数据.

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx] #输入图片index,得到id
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.root_dir, image_info['file_name'])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=image_id)#根据id得到annotation
        anns = self.coco.loadAnns(ann_ids)
        bboxes = []
        masks = []

        for ann in anns:#遍历标注里面的各个物品
            x, y, w, h = ann['bbox']
            bboxes.append([x, y, x + w, y + h])#切换成四角模式.
            mask = self.coco.annToMask(ann)
            masks.append(mask)#添加box信息和mask信息.

        if self.transform:
            image, masks, bboxes = self.transform(image, masks, np.array(bboxes))

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        return image, torch.tensor(bboxes), torch.tensor(masks).float()


def collate_fn(batch):
    images, bboxes, masks = zip(*batch)
    images = torch.stack(images)
    return images, bboxes, masks


class ResizeAndPad:

    def __init__(self, target_size):
        self.target_size = target_size
        self.transform = ResizeLongestSide(target_size)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image, masks, bboxes):#bboxes传入的是0-1之间的float数.
        # Resize image and masks
        og_h, og_w, _ = image.shape
        image = self.transform.apply_image(image)
        masks = [torch.tensor(self.transform.apply_image(mask)) for mask in masks]
        image = self.to_tensor(image)

        # Pad image and masks to form a square
        _, h, w = image.shape
        max_dim = max(w, h)   # 变换图片为一个正方形.
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2

        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)#因为奇偶性,所以是max_dim - w - pad_w, max_dim - h - pad_h
        image = transforms.Pad(padding)(image)# 图像paddign成正方形.
        masks = [transforms.Pad(padding)(mask) for mask in masks] # makspadding成正方形.

        # Adjust bounding boxes
        bboxes = self.transform.apply_boxes(bboxes, (og_h, og_w))#把0-1版本bbox,转化为像素值bbox.
        bboxes = [[bbox[0] + pad_w, bbox[1] + pad_h, bbox[2] + pad_w, bbox[3] + pad_h] for bbox in bboxes]#这个地方也应该跟80行类似的变换. 先放zheli,后续改一下测测效果.

        return image, masks, bboxes


def load_datasets(cfg, img_size): #img_size表示变换后的图片大小.
    transform = ResizeAndPad(img_size)
    train = COCODataset(root_dir=cfg.dataset.train.root_dir,
                        annotation_file=cfg.dataset.train.annotation_file,
                        transform=transform)
    val = COCODataset(root_dir=cfg.dataset.val.root_dir,
                      annotation_file=cfg.dataset.val.annotation_file,
                      transform=transform)
    train_dataloader = DataLoader(train,
                                  batch_size=cfg.batch_size,
                                  shuffle=True,
                                  num_workers=cfg.num_workers,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(val,
                                batch_size=cfg.batch_size,
                                shuffle=False,
                                num_workers=cfg.num_workers,
                                collate_fn=collate_fn)#===test没必要shuffle了.
    return train_dataloader, val_dataloader
