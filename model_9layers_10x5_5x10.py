import torch
import torchvision.transforms.functional
from torch import nn
from PIL import Image
from torchvision import transforms

class Billy(nn.Module):
    def __init__(self):
        super(Billy, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(1,2,(10,5),1,(5,2)),
            nn.MaxPool2d(2),
            nn.Conv2d(2,2,(5,10),1,(2,0)),
            nn.MaxPool2d(2),
            nn.Conv2d(2,4,5),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(20 * 4 * 4, 20),
            nn.Linear(20, 6)
        )

    def forward(self,x):
        x = self.model(x)
        return x

#调试用
#
# billy=Billy()
#
# input = Image.open('dual.jpg')
# input = input.resize((100,100))
# input = input.convert('L')
# trans = transforms.Compose([
#         transforms.ToTensor(),
#         ]
# )
# input = trans(input)
# input = torch.unsqueeze(input,0)
# output = billy(input)
# print(output)
# print(output.shape)
# image = output.cpu().clone()
# image = image.squeeze(0)
# image = torch.reshape(image,[1,-1,image.shape[2]])
# print(image.shape)
# image = torchvision.transforms.functional.to_pil_image(image)
# #image.show()



