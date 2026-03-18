import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn.functional as F


# 1. Настройки и загрузка данных


print(" СРАВНЕНИЕ РОБАСТНОСТИ: ИСХОДНАЯ И ДООБУЧЕННАЯ МОДЕЛИ")


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f" Используется GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f" Используется Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    print(f" GPU/MPS недоступен, используется CPU")

print(f"  Используемое устройство: {device}")

# Трансформации
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Тестовая выборка
testset = CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

print(f" Тестовая выборка: {len(testset)} изображений")



# 2.  Функция PGD-атаки
# -----------------------------
def pgd_attack(model, images, labels, epsilon=8 / 255, alpha=2 / 255, num_iter=7):

    model.eval()

    # Создаём начальное возмущение
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).detach()

    for _ in range(num_iter):

        adv_images = adv_images.clone().detach().requires_grad_(True)

        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        # Сохраняем градиенты
        grad = adv_images.grad

        # Обновление без градиентов
        with torch.no_grad():
            adv_images = adv_images + alpha * grad.sign()
            eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            adv_images = torch.clamp(images + eta, 0, 1)

    model.train()
    return adv_images.detach()



# 3. Функция оценки робастности
# -----------------------------
def evaluate_robustness(model, test_loader, device, attack_fn=None, **attack_kwargs):

    model.eval()
    total = correct = 0

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)


        if attack_fn is not None:
            images = attack_fn(model, images, labels, **attack_kwargs)

        # Вычисляем предсказания без градиентов
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  → {batch_idx + 1}/{len(test_loader)}", end='\r')

    print()
    acc = 100.0 * correct / total
    return acc





original_model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet20',
    pretrained=True,
    trust_repo=True
)
original_model = original_model.to(device)
original_model.eval()



# 5. Загрузка дообученной модели

print(" Загрузка дообученной модели")
adv_model = torch.hub.load(
    'chenyaofo/pytorch-cifar-models',
    'cifar10_resnet20',
    pretrained=False,
    trust_repo=True
)
adv_model.load_state_dict(torch.load('resnet20_adv_trained_pgd.pth'))
adv_model = adv_model.to(device)
adv_model.eval()



# 6. Оценка робастности обеих моделей


print("ОЦЕНКА РОБАСТНОСТИ: ИСХОДНАЯ МОДЕЛЬ")


# Исходная модель: чистые данные
orig_clean = evaluate_robustness(original_model, testloader, device, attack_fn=None)
print(f"  Чистые данные: {orig_clean:.2f}%")

# Исходная модель: слабая PGD (7 итераций)
orig_pgd_weak = evaluate_robustness(
    original_model, testloader, device,
    attack_fn=pgd_attack,
    epsilon=8 / 255,
    alpha=2 / 255,
    num_iter=7
)
print(f"  PGD (ε=8/255, 7 итер.): {orig_pgd_weak:.2f}%")

# Исходная модель: сильная PGD (20 итераций)
orig_pgd_strong = evaluate_robustness(
    original_model, testloader, device,
    attack_fn=pgd_attack,
    epsilon=8 / 255,
    alpha=2 / 255,
    num_iter=20
)
print(f"  PGD (ε=8/255, 20 итер.): {orig_pgd_strong:.2f}%")



print("ОЦЕНКА РОБАСТНОСТИ: ДООБУЧЕННАЯ МОДЕЛЬ")


# Дообученная модель: чистые данные
adv_clean = evaluate_robustness(adv_model, testloader, device, attack_fn=None)
print(f"  Чистые данные: {adv_clean:.2f}%")

# Дообученная модель: слабая PGD (7 итераций)
adv_pgd_weak = evaluate_robustness(
    adv_model, testloader, device,
    attack_fn=pgd_attack,
    epsilon=8 / 255,
    alpha=2 / 255,
    num_iter=7
)
print(f"  PGD (ε=8/255, 7 итер.): {adv_pgd_weak:.2f}%")

# Дообученная модель: сильная PGD (20 итераций)
adv_pgd_strong = evaluate_robustness(
    adv_model, testloader, device,
    attack_fn=pgd_attack,
    epsilon=8 / 255,
    alpha=2 / 255,
    num_iter=20
)
print(f"  PGD (ε=8/255, 20 итер.): {adv_pgd_strong:.2f}%")


# 7. Сводная таблица результатов

print("\n" + "=" * 100)
print("СВОДНАЯ ТАБЛИЦА: СРАВНЕНИЕ РОБАСТНОСТИ МОДЕЛЕЙ")
print("=" * 100)
print(f"{'Модель':<30} | {'Чистые (%)':<12} | {'PGD 7 итер. (%)':<16} | {'PGD 20 итер. (%)':<18}")
print("-" * 100)
print(f"{'Исходная':<30} | {orig_clean:>12.2f} | {orig_pgd_weak:>16.2f} | {orig_pgd_strong:>18.2f}")
print(f"{'Дообученная':<30} | {adv_clean:>12.2f} | {adv_pgd_weak:>16.2f} | {adv_pgd_strong:>18.2f}")
print("=" * 100)

