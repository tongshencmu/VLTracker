import timm
import torch

from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models.vision_transformer import default_cfgs

model_name = 'beitv2_base_patch16_224.in1k_ft_in22k_in1k'
model = timm.create_model(model_name, pretrained=True, img_size=[256, 384])
model.eval()

save_name = model_name.replace('.', '_') + '.pth'
torch.save(model.state_dict(), save_name)

print(model.default_cfg)

# Create transform
config = resolve_data_config(model.pretrained_cfg, model=model)
transform = create_transform(**config)

print(config)
