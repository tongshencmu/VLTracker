"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh

from lib.models.ostrack.clip import TextTransformer
from lib.utils.misc import is_main_process
import open_clip
from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import checkpoint_filter_fn
from timm.models._register import model_entrypoint

from lib.models.ostrack.vit_timm import ViTTIMM


class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, text_encoder=None, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.text_encoder = text_encoder
        if self.text_encoder is not None:
            if self.text_encoder.width != self.backbone.embed_dim:
                self.text_dim_mapper = nn.Linear(
                    self.text_encoder.width, self.backbone.embed_dim)
            else:
                self.text_dim_mapper = nn.Identity()

        self.tokenizer = open_clip.get_tokenizer('ViT-B-16')

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text=None,
                text_embed=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):

        if self.text_encoder is not None:
            assert text is not None or text_embed is not None, "text or text_embed must be provided if text_encoder is not None"
            if text_embed is None and text is not None:
                text = self.tokenizer(text).to(template.device)
                text_embed = self.text_encoder(text)
                text_embed = self.text_dim_mapper(text_embed)
        else:
            text_embed = None

        x, aux_dict = self.backbone(z=template, x=search, text_embed=text_embed,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -
                              self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(
                opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(
        __file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TIMM is False:
        if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
            backbone = vit_base_patch16_224(
                pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
            backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                               ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                               ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                               fusion_loc=cfg.MODEL.BACKBONE.FUSION_LOC,
                                               )
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
            backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                                ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                                ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                                fusion_loc=cfg.MODEL.BACKBONE.FUSION_LOC,
                                                )

            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        else:
            raise NotImplementedError

        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    if cfg.MODEL.BACKBONE.TIMM:

        kwargs = cfg.MODEL.BACKBONE.CFG
        
        model_name = cfg.MODEL.BACKBONE.MODEL_NAME
        model_tag = cfg.MODEL.BACKBONE.MODEL_TAG
        
        backbone = build_model_with_cfg(
            ViTTIMM,
            variant=f'{model_name}.{model_tag}',
            pretrained=False,
            pretrained_filter_fn=checkpoint_filter_fn,
            **kwargs,
        ).cuda()

        hidden_dim = backbone.embed_dim if backbone.num_classes == 0 else backbone.num_classes

        ckpt = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_FILE,
                          map_location="cpu")
        out_dict = checkpoint_filter_fn(ckpt, backbone)
        missing_keys, unexpected_keys = backbone.load_state_dict(
            out_dict, strict=False)
        if is_main_process():
            print('Model Name: ', f'{model_name}.{model_tag}')
            print("Load pretrained image encoder checkpoint from:",
                  cfg.MODEL.BACKBONE.PRETRAINED_FILE)
            print("missing keys:", missing_keys)
            print("unexpected keys:", unexpected_keys)
            print("Loading pretrained TIMM ViT done.")

        if hasattr(backbone, 'customize_vit'):
            backbone.customize_vit()
        else:
            print("No customize_vit function found in the backbone. Skip.")

    if cfg.MODEL.TEXT.USE_TEXT:

        text_encoder = TextTransformer(context_length=cfg.MODEL.TEXT.CONTEXT_LENGTH,
                                       vocab_size=cfg.MODEL.TEXT.VOCAB_SIZE,
                                       width=cfg.MODEL.TEXT.EMBED_DIM,
                                       layers=cfg.MODEL.TEXT.NUM_LAYERS,
                                       heads=cfg.MODEL.TEXT.NUM_HEADS,
                                       embed_cls=cfg.MODEL.TEXT.EMBED_CLS,).cuda()

        if cfg.MODEL.TEXT.PRETRAINED:
            checkpoint = torch.load(cfg.MODEL.TEXT.PRETRAINED_FILE)
            missing_keys, unexpected_keys = text_encoder.load_state_dict(
                checkpoint, strict=False)
            if is_main_process():
                print("Load pretrained text encoder checkpoint from:",
                      cfg.MODEL.TEXT.PRETRAINED_FILE)
                print("missing keys:", missing_keys)
                print("unexpected keys:", unexpected_keys)
                print("Loading pretrained ViT done.")

        text_encoder.requires_grad_(False)

    else:

        text_encoder = None

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        text_encoder=text_encoder,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print("Loading pretrained MAE ViT done.")

    return model
