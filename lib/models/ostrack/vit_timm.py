import torch

from timm.models._builder import build_model_with_cfg
from timm.models.vision_transformer import VisionTransformer, default_cfgs, checkpoint_filter_fn
from timm.layers import resample_abs_pos_embed

class ViTTIMM(VisionTransformer):
    
    def __init__(self,
                 search_img_size,
                 template_img_size,
                 **kwargs):

        super().__init__(**kwargs)
        stride = kwargs['patch_size']
        
        self.search_grid_size = search_img_size // stride
        self.template_grid_size = template_img_size // stride
              
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.pos_embed = self.pos_embed.to(self.device)
        
    def customize_vit(self):
        # resize pos embedding when different size from pretrained weights
        # self.search_pos_embed = resample_abs_pos_embed(
        #     self.pos_embed,
        #     new_size=(self.search_grid_size, self.search_grid_size),
        #     num_prefix_tokens=self.num_prefix_tokens,
        # ).cuda()
        
        # self.template_pos_embed = resample_abs_pos_embed(
        #     self.pos_embed,
        #     new_size=(self.template_grid_size, self.template_grid_size),
        #     num_prefix_tokens=self.num_prefix_tokens,
        # ).cuda()
        
        self.cls_token.requires_grad = False
        self.head.requires_grad = False
        
        # Disable size check in patch embedding module
        self.patch_embed.img_size = None
        
    def forward_features(self, z, x, text_embed=None):
        
        # x = self.patch_embed(x)
        # if self.cls_token is not None:
        #     x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x = x + self.search_pos_embed
        
        # z = self.patch_embed(z)
        # if self.cls_token is not None:
        #     z = z + self.template_pos_embed[:, 1:]
        # else:
        #     z = z + self.template_pos_embed
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = x[:, 1:]
        
        z = self.patch_embed(z)
        z = self._pos_embed(z)
        z = self.patch_drop(z)

        num_prefix_token = 1
            
        if text_embed is not None:
            x = torch.cat([text_embed.unsqueeze(1), z, x], dim=1)
            num_prefix_token += 1 + z.size(1)
        else:
            x = torch.cat([z, x], dim=1)
            num_prefix_token += z.size(1)
            
        x = self.norm_pre(x)
        x = self.blocks(x)
        x = self.norm(x)
        
        return x, num_prefix_token

    def forward_head(self, x, pre_logits: bool = False):
        # if self.global_pool:
        #     x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, z, x, text_embed=None, **kwargs):
        x, num_prefix_tokens = self.forward_features(x=x, z=z, text_embed=text_embed)
        # x = self.forward_head(x)
        # x = x[:, num_prefix_tokens:]
        # B, L, C = x.shape
        # x = x.permute(0, 2, 1).contiguous().reshape(B, C, self.search_grid_size, self.search_grid_size)
        
        z_feat = x[:, :num_prefix_tokens]
        aux_dict = {'template_feat': z_feat}
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
