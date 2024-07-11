from hipt_4k import HIPT_4K
from hipt_model_utils import eval_transforms
import torch
from PIL import Image

model = HIPT_4K(model256_path="../ckpts/vit256_small_dino.pth",model4k_path="../ckpts/vit4k_xs_dino.pth",device256=torch.device('cuda:0'),device4k=torch.device('cuda:0'))
model.eval()

#region = Image.open('image_demo/image_4k.png')
#x = eval_transforms()(region).unsqueeze(dim=0)

slide = Image.open("../../mount_i/treatment_data/120099.svs")
x = eval_transforms()(slide).unsqueeze(dim=0)
out = model.forward(x)
print("output:",out)
