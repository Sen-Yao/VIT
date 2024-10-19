import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import VIT
import CIFAR10
import tqdm

train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
])

trainset = CIFAR10.CIFAR10(root='./CIFAR10_imbalanced', oversampling=False, transform=train_transform)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
class_weights = trainset.class_weights()

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

criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to('cuda'))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    train_loader_tqdm = tqdm.tqdm(trainloader, desc=f"Epoch {epoch+1}/{10}", unit="batch")

    running_loss = 0.0
    for data, target in trainloader:
        data, target = data.to('cuda'), target.to('cuda')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        train_loader_tqdm.set_postfix(loss=running_loss / len(trainloader))

    print("Epoch", epoch, "done. Loss: ", running_loss / len(trainloader))

# save pth
torch.save(model.state_dict(), 'vit_weightloss.pth')



