import torch
import torchvision.models as models

model = models.resnet50(pretrained=True)

input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, input, "resnet50-1.onnx")