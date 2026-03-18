import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# 1. Загрузка модели

print("Загрузка предобученной ResNet-20...")
model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet20',
    pretrained=True,
    trust_repo=True
)
model.eval()
device = torch.device("cpu")
model = model.to(device)
print("Модель загружена.")


# 2. Подготовка данных

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Трансформации — как при обучении модели
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
])

testset = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Берём первые 10 изображений
dataiter = iter(testloader)
images, labels = next(dataiter)
images_vis, labels_vis = images[:10], labels[:10]


# 3. Предсказания

with torch.no_grad():
    outputs = model(images_vis.to(device))
    probs = torch.softmax(outputs, dim=1)
    confidences, predicted = torch.max(probs, dim=1)
    predicted = predicted.cpu()
    confidences = confidences.cpu()


# 4. Функция денормализации

def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    """Возвращает изображение в [0, 1] для корректного отображения."""
    tensor = tensor.clone()  # не меняем оригинал
    for i in range(3):  # для трёх каналов
        tensor[i] = tensor[i] * std[i] + mean[i]
    return torch.clamp(tensor, 0.0, 1.0)


# 5. Визуализация: 10 изображений (2×5)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Примеры работы предобученной ResNet-20 на CIFAR-10", fontsize=14, weight='bold')

for i in range(10):
    ax = axes[i // 5, i % 5]


    img = denormalize(images_vis[i])  # → [3, 32, 32]
    img = img.permute(1, 2, 0).numpy()  # → [32, 32, 3]


    ax.imshow(img, interpolation='nearest')

    # Подпись
    true_label = classes[labels_vis[i].item()]
    pred_label = classes[predicted[i].item()]
    conf = confidences[i].item() * 100

    color = 'green' if predicted[i] == labels_vis[i] else 'red'
    ax.set_title(
        f"Истина: {true_label}\n"
        f"Предсказание: {pred_label}\n"
        f"Уверенность: {conf:.1f}%",
        color=color,
        fontsize=9,
        linespacing=1.3
    )
    ax.axis('off')

plt.tight_layout()
plt.savefig("resnet20_predictions.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.show()


# 6. Точность на всём тестовом наборе

print(" Расчёт точности на всём тестовом наборе (10 000 изображений)")
model.eval()
total = correct = 0

with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

final_accuracy = 100.0 * correct / total
print(f"\n Точность модели на CIFAR-10: {final_accuracy:.2f}%")
