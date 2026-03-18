import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random


# 1. Загрузка предобученной ResNet-20

print("Загрузка предобученной ResNet-20")
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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

testset = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Подвыборка из 1000 изображений
random.seed(42)
subset_1000_indices = random.sample(range(len(testset)), 1000)
subset_loader_1000 = DataLoader(
    testset,
    batch_size=100,
    sampler=torch.utils.data.SubsetRandomSampler(subset_1000_indices)
)

# Выберем 5 случайных изображений для визуализации
indices = random.sample(range(len(testset)), 5)
images_list = [testset[i][0].unsqueeze(0) for i in indices]
labels_list = [testset[i][1] for i in indices]
images = torch.cat(images_list).to(device)
labels = torch.tensor(labels_list).to(device)



# 3. Денормализация

def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    tensor = tensor.clone()
    for i in range(3):
        tensor[i] = tensor[i] * std[i] + mean[i]
    return torch.clamp(tensor, 0.0, 1.0)



# 4. Генерация случайного шума

def generate_random_gaussian_noise(images, std=0.00001):
    """Генерация случайного гауссовского шума"""
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0, 1)


def generate_white_noise(images, magnitude=0.00001):
    """Генерация белого шума (равномерное распределение)"""
    noise = torch.empty_like(images).uniform_(-magnitude, magnitude)
    return torch.clamp(images + noise, 0, 1)



# 5. Применение шума к изображениям

def apply_gaussian_noise(model, images, labels, std=0.00001, device=None):
    """Применение случайного гауссовского шума"""
    return generate_random_gaussian_noise(images, std)


def apply_white_noise(model, images, labels, magnitude=0.00001, device=None):
    """Применение белого шума"""
    return generate_white_noise(images, magnitude)



# 6. Функция оценки

def evaluate_attack(model, attack_fn, data_loader, device, **kwargs):
    model.eval()
    total = correct = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        adv_images = attack_fn(model, images, labels, **kwargs)
        with torch.no_grad():
            preds = model(adv_images).argmax(1)
            correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"  → {batch_idx + 1}/{len(data_loader)}", end='\r')
    print()
    return 100.0 * correct / total



# 7. Оценка на 1000 изображениях

print("\nОценка влияния случайного шума на 1000 изображениях")

# Чистая точность
acc_clean_1000 = evaluate_attack(model, lambda m, x, y, **kw: x, subset_loader_1000, device)
print(f"  Чистые данные: {acc_clean_1000:.2f}%")

# Случайный гауссовский шум
print("  Гауссовский шум (std=0.02)...")
acc_gaussian_1000 = evaluate_attack(
    model,
    apply_gaussian_noise,
    subset_loader_1000,
    device,
    std=0.00001
)

# Белый шум
print("  Белый шум (magnitude=0.02)...")
acc_white_1000 = evaluate_attack(
    model,
    apply_white_noise,
    subset_loader_1000,
    device,
    magnitude=0.00001
)



# 8. Визуализация 5 изображений

def predict(model, images):
    with torch.no_grad():
        out = model(images)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.cpu(), conf.cpu()


# Применяем шумы к 5 изображениям
adv_gaussian = generate_random_gaussian_noise(images, std=0.00001)
adv_white = generate_white_noise(images, magnitude=0.00001)

pred_clean, conf_clean = predict(model, images)
pred_gaussian, conf_gaussian = predict(model, adv_gaussian)
pred_white, conf_white = predict(model, adv_white)


def plot_noise_comparison(clean_imgs, adv_imgs, clean_pred, adv_pred, clean_conf, adv_conf, labels, title, filename):
    """Визуализация сравнения оригинала и зашумлённого изображения"""
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, weight='bold')

    for i in range(5):
        # Оригинал
        img_orig = denormalize(clean_imgs[i]).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_orig, interpolation='nearest')
        true_lbl = classes[labels[i]]
        pred_lbl = classes[clean_pred[i]]
        axes[0, i].set_title(
            f"Оригинал\n{true_lbl} → {pred_lbl}\n{clean_conf[i] * 100:.1f}%",
            fontsize=9, color='green' if clean_pred[i] == labels[i] else 'red'
        )
        axes[0, i].axis('off')

        # Зашумлённое изображение
        img_adv = denormalize(adv_imgs[i]).permute(1, 2, 0).numpy()
        pred_adv_lbl = classes[adv_pred[i]]
        color = 'red' if adv_pred[i] != labels[i] else 'green'
        axes[1, i].imshow(img_adv, interpolation='nearest')
        axes[1, i].set_title(
            f"Зашумлённое\n{pred_adv_lbl}\n{adv_conf[i] * 100:.1f}%",
            fontsize=9, color=color
        )
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


# Визуализация ГАУССОВСКОГО шума
print("\nВизуализация: 5 чистых + 5 с гауссовским шумом")
plot_noise_comparison(
    images,
    adv_gaussian,
    pred_clean,
    pred_gaussian,
    conf_clean,
    conf_gaussian,
    labels,
    "Влияние гауссовского шума (std=0.1) на работу модели",
    "gaussian_noise_comparison.png"
)

# Визуализация БЕЛОГО шума
print("\nВизуализация: 5 чистых + 5 с белым шумом")
plot_noise_comparison(
    images,
    adv_white,
    pred_clean,
    pred_white,
    conf_clean,
    conf_white,
    labels,
    "Влияние белого шума (magnitude=0.1) на работу модели",
    "white_noise_comparison.png"
)


# 9. ТАБЛИЦА РЕЗУЛЬТАТОВ

print("\n" + "=" * 70)
print("Точность модели на CIFAR-10 при случайном шуме (1000 изображений)")
print("=" * 70)
print(f"{'Тип шума':<25} | {'Точность (%)':<12} | {'Падение (%)':<10} | {'Параметр':<12}")
print("-" * 70)
print(f"{'Чистые данные':<25} | {acc_clean_1000:>12.2f} | {'—':<10} | {'—':<12}")
print(f"{'Гауссовский (std=0.1)':<25} | {acc_gaussian_1000:>12.2f} | {acc_clean_1000 - acc_gaussian_1000:>9.2f} | {'std=0.02':<12}")
print(f"{'Белый (magnitude=0.1)':<25} | {acc_white_1000:>12.2f} | {acc_clean_1000 - acc_white_1000:>9.2f} | {'mag=0.02':<12}")
print("=" * 70)
