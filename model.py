import torch.nn as nn
import timm 
from config import * 


# Using efficientnet_b0

class SignEfficientNet(nn.Module):
  def __init__(self, num_classes = NUM_CLASSES):
    super().__init__()

    self.model = timm.create_model('efficientnet_b0', pretrained = True)


    ## Freezing backbone
    for param in self.model.parameters():
            param.requires_grad = False

    in_features = self.model.classifier.in_features 

    self.model.classifier = nn.Sequential(
      nn.Dropout(p = 0.3),
      nn.Linear(in_features, num_classes)
    )

    for param in self.model.classifier.parameters():
            param.requires_grad = True

  def unfreeze_backbone(self):
        """
        Fine-tuning stage:
        unfreeze all layers
        """

        for param in self.model.parameters():
            param.requires_grad = True

  def forward(self, x):
    return self.model(x)