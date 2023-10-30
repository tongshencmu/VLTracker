import torch

import timm.models._builder import build_model_with_cfg
import timm.models.eva import Eva
from timm.layers import resample_abs_pos_embed

class EvaTrack(Eva):
    
    def __init__(self, 
                 search_img_size,
                 template_img_size,
                 use_class_token=False
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_class_token = use_class_token
        
        self.search_grid_size = search_img_size // stride
        self.template_grid_size = template_img_size // stride
        
    def customize_vit(self):
        
        self.search_pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(self.search_grid_size, self.search_grid_size),
            num_prefix_tokens=self.num_prefix_tokens,
        ).cuda()
        
        self.template_pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(self.template_grid_size, self.template_grid_size),
            num_prefix_tokens=self.num_prefix_tokens,
        ).cuda()
        
        if not self.use_class_token:
            self.cls_token.requires_grad = False
            
        # self.head.requires_grad = False
        
        # support dynamic image size
        self.patch_embed.img_size = None
        
    def forward_features(self, z, x, text_embed=None):
        
        x = self.patch_embed(x)
        z = self.patch_embed(z)
        
        x, rot_pos_embed = self._pos_embed(x)
        z, rot_pos_embed_z = self._pos_embed(z)
        
        x = torch.cat([z, x[:, 1:]], dim=1)
        rot_pos_embed = torch.cat([rot_pos_embed_z, rot_pos_embed], dim=1)
        
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
            
        x = self.norm(x)
        return x
    
    def forward_head(self, x):
        
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)
    
    def forward(self, z, x, text_embed=None, **kwargs):
        
        x = self.forward_features(x=x, z=z, text_embed=text_embed)
        x = self.forward_head(x)
        
        aux_dict = {}
        return x, aux_dict