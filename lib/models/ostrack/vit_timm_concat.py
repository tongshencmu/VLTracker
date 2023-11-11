import torch

from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer, default_cfgs, checkpoint_filter_fn
from timm.layers import resample_abs_pos_embed

import torch.nn.functional as F

class ViTTIMMConcat(VisionTransformer):
    
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
              
    def forward(self, z, x, text_embed=None, **kwargs):
        
        # Empty
        empty_z = torch.zeros_like(z).to(z.device)
        input_img = torch.cat([z, empty_z], axis=2)
        input_img = torch.cat([input_img, x], axis=3)
        
        # Same
        # z = F.interpolate(z, scale_factor=2, mode='bilinear', align_corners=False)
        # input_img = torch.cat([input_img, z], axis=3)
        
        # Trident
        # input_img = torch.cat(z, axis=2)
        # input_img = torch.cat([input_img, x], axis=3)
        
        x = self.forward_features(input_img)
        
        x = x[:, self.num_prefix_tokens:]
        
        B = x.shape[0]
        feature_size = [self.patch_embed.img_size[0] // self.patch_embed.patch_size[0],
                        self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]]
        x = x.reshape(B, feature_size[0], feature_size[1], -1)
        x = x[:, :, self.template_grid_size:, :]
        x = x.flatten(1, 2)

        aux_dict = {}
        return x, aux_dict
        
if __name__ == '__main__':
    
    # def func_kwargs(arg1, **kwargs):
    #     print('arg1 =', arg1)
    #     print('kwargs =', kwargs)

    # func_kwargs(**{'arg1': 'one', 'arg2': 'two', 'arg3': 'three'})

    # args_dict = {}
    # args_dict = {'template_img_size': 128, 'search_img_size': 256}
    # args_dict = {'pre_norm': True, 'template_img_size': 128, 'search_img_size': 256, 'class_token': True, 
    # 'use_class_token': False, 'num_classes': 512, 'in_chans': 3, 'img_size': (224, 224)}
    # model = VisionTransformer().cuda()
    # a = torch.randn(1, 3, 224, 224).cuda()
    # x = model(a)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # x.sum().backward()
    # optimizer.zero_grad()
    # optimizer.step()
    
    # model = ViTFusion(**args_dict).cuda()
    # model.customize_vit()
    # a = torch.randn(1, 3, 128, 128).cuda()
    # b = torch.randn(1, 3, 256, 256).cuda()
    # x, aux_dict = model(a, b)
    # print(model.template_pos_embed.shape, model.search_pos_embed.device)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    # x.sum().backward()
    # for name, param in model.named_parameters():
    #     if param.grad is not None:
    #         print(name, param.device, param.shape)
    #         print(name, param.grad.device, param.grad.shape)
    # optimizer.zero_grad()
    # optimizer.step()
    # print(model)
    
    kwargs = dict(pre_norm=True,
                  template_img_size=128,
                  search_img_size=256,
                  stride=16,
                  use_class_token=False)
    model_name = 'vit_base_patch16_clip_224'
    model_tag = 'laion2b'

    backbone = build_model_with_cfg(
        ViTTIMM,
        variant=f'{model_name}.{model_tag}',
        pretrained=False,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs,
    ).cuda()

    hidden_dim = backbone.embed_dim

    ckpt = torch.load('vit-b-16-laion-2b.pth', map_location="cpu")
    out_dict = checkpoint_filter_fn(ckpt, backbone)
    missing_keys, unexpected_keys = backbone.load_state_dict(
        out_dict, strict=False)
    print('Model Name: ', f'{model_name}.{model_tag}')
    print("missing keys:", missing_keys)
    print("unexpected keys:", unexpected_keys)
    print("Loading pretrained TIMM ViT done.")

    backbone.customize_vit()
