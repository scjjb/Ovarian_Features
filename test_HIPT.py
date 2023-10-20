## model testing code folloing https://github.com/mahmoodlab/HIPT/blob/a9b5bb8d159684fc4c2c497d68950ab915caeb7e/HIPT_4K/HIPT_4K%20Inference%20%2B%20Attention%20Visualization.ipynb#L26
## correct out given in https://github.com/mahmoodlab/HIPT/issues/56

from HIPT_4K.hipt_4k import HIPT_4K
from HIPT_4K.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from HIPT_4K.hipt_heatmap_utils import *
import argparse

parser = argparse.ArgumentParser(description='Configurations for HIPT feature extraction')
parser.add_argument('--hardware',type=str, choices=['DGX','PC'], default='DGX',help='sets amount of CPU and GPU to use per experiment')
args = parser.parse_args()

if args.hardware == "PC":
    pretrained_weights256 = 'HIPT_4K/ckpts/vit256_small_dino.pth'
    pretrained_weights4k = 'HIPT_4K/ckpts/vit4k_xs_dino.pth'
else:
    pretrained_weights256 ="/mnt/results/Checkpoints/vit256_small_dino.pth"
    pretrained_weights4k = "/mnt/results/Checkpoints/vit4k_xs_dino.pth"

device256 = torch.device('cuda')
device4k = torch.device('cuda')

model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ",device)
model.eval()

region = Image.open('HIPT_4K/image_demo/image_4k.png')
x = eval_transforms()(region).unsqueeze(dim=0)
out = model.forward(x)

expected_out = torch.tensor([[0.8896,-2.1130,0.4011,1.9388,2.2679,-0.2919,-2.8318,3.3083,
    -2.5549,-1.0718,2.4532,0.3009,-2.7087,1.0475,0.4862,-0.9086,
    -0.6283,-1.4109,-1.7757,-0.4216,-2.2767,-0.0307,3.0037,-0.7022,
    -2.2229,-2.5973,4.2466,-2.3519,1.0857,-0.5460,-2.3129,2.3446,
    -3.0198,-2.6937,1.9349,-0.4484,0.0817,-0.6997,-0.2162,-0.9967,
    2.8000,-5.1581,2.1064,-0.2916,1.1988,0.3805,4.4717,4.1056,
    0.2514,-1.3006,-4.8284,-0.1595,1.9322,1.7319,-6.2168,-3.1303,
    -2.8676,3.3709,0.2881,-2.3995,2.5332,1.3674,-0.2954,-0.7680,
    2.0302,1.5359,1.7415,3.3354,2.2949,1.5521,-0.8878,-1.7468,
    -2.7638,-2.0117,-0.6663,-3.2415,-0.0986,-0.0882,2.2837,4.6560,
    -0.2273,1.5000,6.0354,2.5131,-0.9898,-4.1885,-2.4678,0.2505,
    -1.7514,0.2658,1.1252,-3.6720,-1.4542,-3.0706,-0.1247,2.9184,
    3.1167,2.7436,-5.0488,-0.0320,-0.1860,-0.1426,0.3627,0.0395,
    -0.6932,0.4822,1.9542,-1.8605,1.4506,1.4573,1.0065,1.4675,
    2.7202,3.6223,0.3807,0.4021,-3.9630,0.0964,-1.6146,-1.2587,
    -1.3076,1.1944,-1.5511,4.7308,-0.0444,4.8179,2.0997,1.1377,
    1.3265,0.2599,2.8389,-4.3338,4.6401,1.6044,2.7752,2.9454,
    -1.8182,-2.2276,-1.6382,-1.5304,2.0726,-0.6284,0.8725,-4.1951,
    1.2878,-0.0490,-1.1738,0.3888,1.4261,1.8519,3.9931,1.1734,
    -2.4811,0.5972,-3.3668,-0.0365,-2.2376,3.1537,3.0984,2.0863,
    -1.1236,-0.7329,-0.9192,-3.4123,-1.0592,-1.0717,-2.1983,-3.0891,
    -0.2500,-3.1052,-0.3217,-0.0544,6.5555,3.3587,-2.7746,-2.2714,
    2.2318,-2.9227,2.5831,-4.2082,2.9219,-0.4439,-2.7881,0.6900,
    -0.7225,-3.2197,0.5538,-0.5984,0.9696,-2.2826,-0.3154,2.4052]],
    device='cuda:0')

if torch.equal(torch.round(out,decimals=2), torch.round(expected_out,decimals=2)):
    print("Test passed - expected features extracted")
else: 
    diff = torch.eq(torch.round(out,decimals=2), torch.round(expected_out,decimals=2))
    similarity = (100*torch.sum(diff)/torch.numel(diff)).item()
    print("Test failed - expected feature similarity {}%".format(round(similarity,2)))
