## model testing code folloing https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/HIPT_4K%20Inference%20%2B%20Attention%20Visualization.ipynb#L26
## correct out given in https://github.com/mahmoodlab/HIPT/issues/56

from HIPT_4K.hipt_4k import HIPT_4K
from HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from HIPT_4K.hipt_heatmap_utils import *

light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)

pretrained_weights256 = 'HIPT_4K/ckpts/vit256_small_dino.pth'
pretrained_weights4k = 'HIPT_4K/ckpts/vit4k_xs_dino.pth'
device256 = torch.device('cuda:0')
device4k = torch.device('cuda:0')

### ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

### ViT_256 + ViT_4K loaded into HIPT_4K API
model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
model.eval()


region = Image.open('HIPT_4K/image_demo/image_4k.png')
x = eval_transforms()(region).unsqueeze(dim=0)
print('Input Shape:', x.shape)
out = model.forward(x)
print('Output Shape:', out.shape)

print("out:",out)
