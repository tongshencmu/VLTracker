import torch

from timm.models._builder import build_model_with_cfg
from timm.models.eva import Eva
from timm.layers import resample_abs_pos_embed

class EvaConcat(Eva):
    
    def __init__(self,
                 search_img_size,
                 template_img_size,
                 use_class_token=False,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.use_class_token = use_class_token
        stride = kwargs['patch_size']
        
        self.search_grid_size = search_img_size // stride
        self.template_grid_size = template_img_size // stride
        
    def forward_features(self, z, x, text_embed=None):
        
        empty_z = torch.zeros_like(z).to(z.device)
        input_img = torch.cat([z, empty_z], axis=2)
        input_img = torch.cat([input_img, x], axis=3)
        
        x = self.patch_embed(input_img)
        x, rot_pos_embed = self._pos_embed(x)

        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
            
        x = self.norm(x)
        return x
    
    def forward(self, z, x, text_embed=None, **kwargs):
        
        x = self.forward_features(x=x, z=z, text_embed=text_embed)
        
        x = x[:, self.num_prefix_tokens:]
        
        B = x.shape[0]
        feature_size = [self.patch_embed.grid_size[0],
                        self.patch_embed.grid_size[1]]
        x = x.reshape(B, feature_size[0], feature_size[1], -1)
        x = x[:, :, self.template_grid_size:, :]
        x = x.flatten(1, 2)
        
        aux_dict = {}
        return x, aux_dict
