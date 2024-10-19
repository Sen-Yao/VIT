model_pth_path = 'vit_oversample.pth'

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import VIT
import CIFAR10

test_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])

testset = CIFAR10.CIFAR10(root='./CIFAR10_balance', oversampling=False, transform=test_transform)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

model = VIT.VisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=2,
    num_heads=4,
    hidden_dim=64,
    mlp_dim=128,
    dropout=0.1,
    attention_dropout=0.1,
    num_classes=10
).to('cuda')

model.load_state_dict(torch.load(model_pth_path))

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in testloader:
        data, target = data.to('cuda'), target.to('cuda')
        output = model(data)
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == torch.max(target, 1)[1]).sum().item()

print('total images: %d' % total)
print('correct images: %d' % correct)
print('accuracy: %f' % (correct / total))
