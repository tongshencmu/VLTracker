import torch
from PIL import Image
import sys

# sys.path.append("/ocean/projects/ele220002p/tongshen/code/vl_tracking/latest/open_clip/src/")
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')

image = preprocess(Image.open("CLIP.png")).unsqueeze(0)
text = tokenizer(["a diagram", "a dog", "a cat", 'skateboard moving on the road under a man in red coat'])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    print(text_features.shape)
    print(text_features[3])
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]