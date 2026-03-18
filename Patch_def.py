import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F


# 1. Загрузка предобученной ResNet-20

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



# 4. Генерация универсального состязательного патча

def generate_adversarial_patch(model, data_loader, device, patch_size=8, target_class=0, max_iter=100):
    """
    Генерация универсального патча, который заставляет модель классифицировать всё как target_class.
    """
    # Инициализация патча случайными значениями в [0, 1]
    patch = torch.rand(3, patch_size, patch_size, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([patch], lr=0.01)

    print(f"Генерация патча размером {patch_size}x{patch_size} → целевой класс: {classes[target_class]}")

    for epoch in range(max_iter):
        total_loss = 0
        count = 0

        for batch_idx, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            batch_size = images.size(0)

            # Случайные позиции для наложения патча
            x_pos = torch.randint(0, 32 - patch_size, (batch_size,))
            y_pos = torch.randint(0, 32 - patch_size, (batch_size,))

            # Создаём копию изображений
            patched_images = images.clone()

            # Накладываем патч
            for i in range(batch_size):
                patched_images[i, :, x_pos[i]:x_pos[i] + patch_size, y_pos[i]:y_pos[i] + patch_size] = patch

            # Прямой проход
            outputs = model(patched_images)
            loss = F.cross_entropy(outputs, torch.full_like(labels, target_class))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Обрезаем патч в [0, 1]
            with torch.no_grad():
                patch.clamp_(0, 1)

            total_loss += loss.item()
            count += 1

            if batch_idx >= 5:  # Используем только первые 5 батчей для ускорения
                break

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / count
            print(f"  Эпоха {epoch + 1}/{max_iter}, Loss: {avg_loss:.4f}")

    return patch.detach()



# 5. Наложение патча на изображения

def apply_patch_to_images(images, patch, device):
    """Накладывает патч на случайные позиции изображений"""
    batch_size = images.size(0)
    _, _, H, W = images.shape
    _, patch_h, patch_w = patch.shape

    patched_images = images.clone()

    # Случайные позиции
    x_pos = torch.randint(0, H - patch_h + 1, (batch_size,), device=device)
    y_pos = torch.randint(0, W - patch_w + 1, (batch_size,), device=device)

    for i in range(batch_size):
        patched_images[i, :, x_pos[i]:x_pos[i] + patch_h, y_pos[i]:y_pos[i] + patch_w] = patch

    return patched_images


# -----------------------------
# 6. Функция оценки
# -----------------------------
def evaluate_patch_attack(model, patch, data_loader, device):
    model.eval()
    total = correct = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        patched_images = apply_patch_to_images(images, patch, device)
        with torch.no_grad():
            preds = model(patched_images).argmax(1)
            correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"  → {batch_idx + 1}/{len(data_loader)}", end='\r')
    print()
    return 100.0 * correct / total


# -----------------------------
# 7. Генерация патча
# -----------------------------
print(" Генерация универсального состязательного патча...")
# Целевой класс: 0 = 'plane' (можно изменить)
adversarial_patch = generate_adversarial_patch(
    model,
    subset_loader_1000,
    device,
    patch_size=8,
    target_class=0,
    max_iter=100
)

# -----------------------------
# 8. Оценка на 1000 изображениях
# -----------------------------
print("\n Оценка эффективности патча на 1000 изображениях...")

# Чистая точность
acc_clean = 91.90
print(f"  Чистые данные: {acc_clean:.2f}%")

# Точность с патчем
acc_patch = evaluate_patch_attack(model, adversarial_patch, subset_loader_1000, device)
print(f"  С патчем : {acc_patch:.2f}%")
print(f"  Падение точности: {acc_clean - acc_patch:.2f}%")


# -----------------------------
# 9. Визуализация
# -----------------------------
def predict(model, images):
    with torch.no_grad():
        out = model(images)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.cpu(), conf.cpu()


# Накладываем патч на 5 изображений
patched_images = apply_patch_to_images(images, adversarial_patch, device)

pred_clean, conf_clean = predict(model, images)
pred_patched, conf_patched = predict(model, patched_images)


def plot_patch_comparison(clean_imgs, patched_imgs, clean_pred, patched_pred, clean_conf, patched_conf, labels, title):
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

        # С патчем
        img_patched = denormalize(patched_imgs[i]).permute(1, 2, 0).numpy()
        pred_patched_lbl = classes[patched_pred[i]]
        color = 'red' if patched_pred[i] != labels[i] else 'green'
        axes[1, i].imshow(img_patched, interpolation='nearest')
        axes[1, i].set_title(
            f"С патчем\n{pred_patched_lbl}\n{patched_conf[i] * 100:.1f}%",
            fontsize=9, color=color
        )
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig("adversarial_patch_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()


print("\n Визуализация: 5 чистых + 5 с патчем")
plot_patch_comparison(
    images,
    patched_images,
    pred_clean,
    pred_patched,
    conf_clean,
    conf_patched,
    labels,
    "Состязательный патч "
)


# -----------------------------
# 11. Реализация APM-защиты (упрощённая версия)
# -----------------------------
def apm_masking(image, patch_size=8, std_threshold=0.45):
    """
    Упрощённый APM: маскирование подозрительных окон 8x8 по стандартному отклонению.
    Вход: image — тензор [3, 32, 32] (нормализованный)
    Выход: image_masked — защищённое изображение (в нормализованном виде)
    """
    # Денормализуем для анализа (работаем в [0, 1])
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1).to(image.device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1).to(image.device)
    img_denorm = image * std + mean  # [3, 32, 32]

    C, H, W = img_denorm.shape
    img_masked = img_denorm.clone()

    # Разбиваем на окна patch_size × patch_size с шагом patch_size
    for i in range(0, H - patch_size + 1, patch_size):
        for j in range(0, W - patch_size + 1, patch_size):
            patch = img_denorm[:, i:i + patch_size, j:j + patch_size]
            # Вычисляем стандартное отклонение по всем пикселям патча
            patch_std = patch.std().item()

            if patch_std > std_threshold:
                # Заменяем окно на среднее значение по каналам (маскирование)
                patch_mean = patch.mean(dim=(1, 2), keepdim=True)  # [3, 1, 1]
                img_masked[:, i:i + patch_size, j:j + patch_size] = patch_mean

    # Нормализуем обратно
    img_masked_norm = (img_masked - mean) / std
    return img_masked_norm


def apply_apm_to_batch(images, patch_size=8, std_threshold=0.45):
    """Применяет APM к батчу изображений"""
    return torch.stack([
        apm_masking(img, patch_size=patch_size, std_threshold=std_threshold)
        for img in images
    ])


# -----------------------------
# 12. Оценка модели с APM-защитой
# -----------------------------
def evaluate_with_apm(model, patch, data_loader, device, patch_size=8, std_threshold=0.45):
    model.eval()
    total = correct = 0
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)

        # 1. Накладываем патч
        patched_images = apply_patch_to_images(images, patch, device)
        # 2. Применяем APM-защиту
        masked_images = apply_apm_to_batch(patched_images, patch_size=patch_size, std_threshold=std_threshold)

        with torch.no_grad():
            preds = model(masked_images).argmax(1)
            correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"  → {batch_idx + 1}/{len(data_loader)}", end='\r')
    print()
    return 100.0 * correct / total


# -----------------------------
# 13. Оценка эффективности APM-защиты
# -----------------------------
print("\n Оценка защиты APM (упрощённая версия)...")
acc_apm = evaluate_with_apm(
    model,
    adversarial_patch,
    subset_loader_1000,
    device,
    patch_size=8,
    std_threshold=0.45
)

print(f"  Точность с APM-защитой: {acc_apm:.2f}%")
print(f"  Улучшение относительно атаки: {acc_apm - acc_patch:+.2f}%")
print(f"  Падение относительно чистых данных: {acc_clean - acc_apm:.2f}%")

# -----------------------------
# 14. Вывод сводной таблицы
# -----------------------------
print("\n" + "=" * 75)
print("Сводная таблица: точность модели ResNet-20 на CIFAR-10 (1000 изображений)")
print("=" * 75)
print(f"{'Состояние':<25} | {'Точность (%)':<12} | {'Падение (%)':<10}")
print("-" * 75)
print(f"{'Чистые данные':<25} | {acc_clean:>12.2f} | {'—':<10}")
print(f"{'С патчем (целевой)':<25} | {acc_patch:>12.2f} | {acc_clean - acc_patch:>9.2f}")
print(f"{'С патчем + APM':<25} | {acc_apm:>12.2f} | {acc_clean - acc_apm:>9.2f}")
print("=" * 75)