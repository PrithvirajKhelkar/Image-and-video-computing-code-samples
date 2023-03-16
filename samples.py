# Problem set 1 samples


def ncc(img_a: np.array,img_b: np.array)-> float:
    """
     Compute the normalized cross-correlation between two color channel images
     and return the matching score.
     
     :param img_a: the first image, which is a 341x396 numpy array.
     :param img_b: the second image, which is a 341x396 numpy array.
     
     :return: the normalized cross-correlation score.
    """  
     
  
    ncc=0
    rows = max(img_a.shape[0], img_b.shape[0])
    cols = max(img_a.shape[1], img_b.shape[1])
    img_a_padded = np.zeros((rows, cols), dtype=np.uint8)
    img_b_padded = np.zeros((rows, cols), dtype=np.uint8)
    img_a_padded[:img_a.shape[0], :img_a.shape[1]] = img_a
    img_b_padded[:img_b.shape[0], :img_b.shape[1]] = img_b
    img_a_padded = img_a_padded - np.mean(img_a_padded)
    img_b_padded = img_b_padded - np.mean(img_b_padded)
    img_a_padded = img_a_padded / np.std(img_a_padded)
    img_b_padded = img_b_padded / np.std(img_b_padded)
    ncc =  np.dot(img_a_padded.flatten(), img_b_padded.flatten()) / (np.linalg.norm(img_a_padded) * np.linalg.norm(img_b_padded))
    return ncc



def recursive_displacement(b:np.array,g:np.array,r:np.array,i:int=1)->[np.array, np.array, np.array, np.array]  :
    """
        A recursive function to colorize 3 channels in an image, given they arent aligned.

        'align_imga_to_imgb()' is a function that takes two channels and aligns them, returning the displacement.
        It slides a window over the image, displacing it with a small value and adds up the ncc scores. The displacement with the smallest score is returned.
    """
    displacements=np.zeros((3, 2))
    aligned_b, aligned_g, aligned_r = None, None, None

    assert b.shape == g.shape == r.shape

    if i < 5:
      resized_b = b[::2, ::2]
      resized_g = g[::2, ::2]
      resized_r = r[::2, ::2]
      _, _, _, displacements = recursive_displacement(resized_b, resized_g, resized_r, i+1)
      displacements = displacements*2
      b = shift(b, displacements[0][0], displacements[0][1])
      g = shift(g, displacements[1][0], displacements[1][1])
      r = shift(r, displacements[2][0], displacements[2][1])

    wd_size = 15 if i == 5 else 2
    aligned_b, dis_b = align_imga_to_imgb(b,g, wd_size)
    aligned_r, dis_r = align_imga_to_imgb(r,g, wd_size)
    aligned_g, dis_g = g, [0,0]
    displacements += np.array([dis_b, dis_g, dis_r])
    
    return aligned_b, aligned_g, aligned_r, displacements

#---------------------------------------------------------------------------------------------#

# Problem set 2 samples

import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """
        A model to classify images in the CIFAR-10 dataset.

        Given the following architecture, improved on it to get a better score.

        Layer No.	Layer Type	Kernel Size	Input Dim	Output Dim	Input Channels	Output Channels
        1	conv2d	5	32	28	3	6
        2	relu	-	28	28	6	6
        3	maxpool2d	2	28	14	6	6
        4	conv2d	5	14	10	6	16
        5	relu	-	10	10	16	16
        6	maxpool2d	2	10	5	16	16
        7	linear	-	1	1	400	200
        8	relu	-	1	1	200	200
        9	linear	-	1	1	200	10

    """
    def __init__(self):
        super(BaseNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride = 1, padding = 1)
        self.norm = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.LeakyReLU(inplace = True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace = True)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace = True)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace = True)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(inplace = True)
        )

        self.reg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=10)
        )
  
    def forward(self, x):
      
        # TODO: define your model here

        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.reg(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


class SurfaceNormalEstimation(nn.Module):
    """
        Model architecture for surface normal estimation for images.

        Tried a few approaches, settled on this. Given ablation table:

        Name and description | Mean Median 11.25 deg 22.5 deg 30 deg L1 Loss
        Resnet-18 modified (more layers) with skip-connections(DeepLabv3+) with angular error loss | 34.4 | 29.9 | 18 | 38.6 | 50.2 | 1.31 |
        Resnet-18 modified (more layers) with skip-connections(DeepLabv3+) with cosine-similarity loss | 36.5 | 31.3 | 17.1 | 37.2 | 48.2 | 0.8 |
        Resnet-18 modified with skip-connections(DeepLabv3+) with cosine-similarity loss | 35.7 | 30.3 | 17.9 | 38.4 | 49.6 | 1.12 |
        Resnet-18 modified with skip-connections(DeepLabv3+) with L1 Loss | 39.2 | 35.5 | 16.7 | 33.3 | 43.2 | 0.26
        Resnet-18 modified | 36.8 | 32.5 | 17.3 | 36 | 46.8  | 0.42 |


    """
    def __init__(self, num_classes=3):
        super(SurfaceNormalEstimation, self).__init__()
        
        # Use ResNet as encoder
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.maxpool = nn.Identity()

        # Decoder
        self.conv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.ConvTranspose2d(256 + 256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.ConvTranspose2d(128 + 128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.ConvTranspose2d(64 + 64, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1)

        # Skip connections
        self.skip_conv1 = nn.Conv2d(256, 256, kernel_size=1)
        self.skip_conv2 = nn.Conv2d(128, 128, kernel_size=1)
        self.skip_conv3 = nn.Conv2d(64, 64, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.resnet.conv1(x)
        x1 = self.resnet.bn1(x1)
        x1 = self.resnet.relu(x1)
        x1 = self.resnet.layer1(x1)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        # Decoder
        x = self.conv1(x4)
        x = self.bn1(x)
        x = F.relu(x)

        # Skip connection from layer3
        x = torch.cat([x, self.skip_conv1(x3)], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Skip connection from layer2
        x = torch.cat([x, self.skip_conv2(x2)], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # Skip connection from layer1
        x = torch.cat([x, self.skip_conv3(x1)], dim=1)
        x = self.conv4(x)
        
        return x

  
