import torch
from models.model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device --> {device}")
data = torch.rand(1,3,640,640).to(device)
net = YOLOv7().to(device)
output = net(data)
print(f"Type : {type(output)}, {len(output)}, {output[0].shape}, {output[1].shape}, {output[2].shape}, {type(output[0])}")