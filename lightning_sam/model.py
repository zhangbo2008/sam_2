import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything import SamPredictor


class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):
        self.model = sam_model_registry[self.cfg.model.type](checkpoint=self.cfg.model.checkpoint)
        self.model.train()#=======根据配置文件进行模型freeze
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            for param in self.model.prompt_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.mask_decoder:
            for param in self.model.mask_decoder.parameters():
                param.requires_grad = False

    def forward(self, images, bboxes):
        _, _, H, W = images.shape
        image_embeddings = self.model.image_encoder(images)
        pred_masks = []
        ious = []#========图像编码,然后给一个prompt: 一个点或者一个box, 然后用mask_decoder网络给出mask和iou结果.
        for embedding, bbox in zip(image_embeddings, bboxes):#=======对于coco数据集.我们把图像跟bbox的gt喂给model,
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )#prompt_encoder. boxes喂入我们的box. 结果是系数编码和稠密编码

            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0), #我们对于每一个图像我们增加第一个维度作为batch_size
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )#这个网络负责生成maks #

            masks = F.interpolate(#=====最后把低分辨率的masks,插值得到原始图片的maks#具体里面一些参数还是要把sam代码看懂.
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)

        return pred_masks, ious

    def get_predictor(self):
        return SamPredictor(self.model)
