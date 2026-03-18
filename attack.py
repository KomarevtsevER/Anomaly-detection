import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import foolbox as fb


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


random.seed(42)
subset_500_indices = random.sample(range(len(testset)), 500)
subset_loader_500 = DataLoader(
    testset,
    batch_size=100,
    sampler=torch.utils.data.SubsetRandomSampler(subset_500_indices)
)

# Выберем 5 случайных изображений для визуализации (seed=42)
indices = random.sample(range(len(testset)), 5)
images_list = [testset[i][0].unsqueeze(0) for i in indices]
labels_list = [testset[i][1] for i in indices]
images = torch.cat(images_list).to(device)
labels = torch.tensor(labels_list).to(device)



def denormalize(tensor, mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)):
    tensor = tensor.clone()
    for i in range(3):
        tensor[i] = tensor[i] * std[i] + mean[i]
    return torch.clamp(tensor, 0.0, 1.0)

def fgsm_attack(model, images, labels, epsilon=8/255):
    images = images.clone().detach().requires_grad_(True)
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    perturbation = epsilon * images.grad.sign()
    adv_images = images + perturbation
    return torch.clamp(adv_images, 0, 1).detach()

def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_iter=20):
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)
    for _ in range(num_iter):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = torch.nn.functional.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        perturbation = alpha * adv_images.grad.sign()
        adv_images = adv_images + perturbation
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = (images + eta).detach()
    return torch.clamp(adv_images, 0, 1)

def deepfool_attack(model, images, labels, overshoot=0.02, steps=50):
    images_denorm = denormalize(images)
    fmodel = fb.PyTorchModel(
        model,
        bounds=(0, 1),
        device=device,
        preprocessing=dict(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
            axis=-3
        )
    )
    attack = fb.attacks.L2DeepFoolAttack(steps=steps, candidates=10, overshoot=overshoot)
    _, adv_images_denorm, _ = attack(fmodel, images_denorm, labels, epsilons=None)
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
    adv_images_norm = (adv_images_denorm - mean) / std
    return adv_images_norm.detach()


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
        if (batch_idx + 1) % 5 == 0:  # чаще обновляем прогресс (500/100 = 5 батчей)
            print(f"  → {batch_idx + 1}/{len(data_loader)}", end='\r')
    print()
    return 100.0 * correct / total


# 7. Оценка на 500 изображениях

print("\n Оценка уязвимости на 500 изображениях ")

# Чистая точность на 500
acc_clean_500 = evaluate_attack(model, lambda m, x, y, **kw: x, subset_loader_500, device)
print(f"  Чистые данные: {acc_clean_500:.2f}%")

# Атаки
print("  FGSM (ε=8/255)...")
acc_fgsm_500 = evaluate_attack(model, fgsm_attack, subset_loader_500, device, epsilon=8/255)

print("  PGD (ε=8/255, 20 шагов)...")
acc_pgd_500 = evaluate_attack(model, pgd_attack, subset_loader_500, device, epsilon=8/255, alpha=2/255, num_iter=20)

print("  DeepFool...")
acc_df_500 = evaluate_attack(model, deepfool_attack, subset_loader_500, device, overshoot=0.02, steps=50)


# Визуализация 5 изображений

adv_fgsm = fgsm_attack(model, images, labels, epsilon=8/255)
adv_pgd = pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, num_iter=20)
adv_df = deepfool_attack(model, images, labels, overshoot=0.02, steps=50)

def predict(model, images):
    with torch.no_grad():
        out = model(images)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.cpu(), conf.cpu()

pred_clean, conf_clean = predict(model, images)
pred_fgsm, conf_fgsm = predict(model, adv_fgsm)
pred_pgd, conf_pgd = predict(model, adv_pgd)
pred_df, conf_df = predict(model, adv_df)

def plot_group(clean_imgs, adv_imgs, clean_pred, adv_pred, clean_conf, adv_conf, labels, title):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, weight='bold')
    for i in range(5):
        img_orig = denormalize(clean_imgs[i]).permute(1, 2, 0).numpy()
        axes[0, i].imshow(img_orig, interpolation='nearest')
        true_lbl = classes[labels[i]]
        pred_lbl = classes[clean_pred[i]]
        axes[0, i].set_title(
            f"Оригинал\n{true_lbl} → {pred_lbl}\n{clean_conf[i] * 100:.1f}%",
            fontsize=9, color='green' if clean_pred[i] == labels[i] else 'red'
        )
        axes[0, i].axis('off')

        img_adv = denormalize(adv_imgs[i]).permute(1, 2, 0).numpy()
        pred_adv_lbl = classes[adv_pred[i]]
        color = 'red' if adv_pred[i] != labels[i] else 'green'
        axes[1, i].imshow(img_adv, interpolation='nearest')
        axes[1, i].set_title(
            f"Атакованное\n{pred_adv_lbl}\n{adv_conf[i] * 100:.1f}%",
            fontsize=9, color=color
        )
        axes[1, i].axis('off')
    plt.tight_layout()
    filename = f"group_{title.split()[0].lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

print("\n Визуализация: 5 чистых + 5 FGSM")
plot_group(images, adv_fgsm, pred_clean, pred_fgsm, conf_clean, conf_fgsm, labels, "FGSM (ε=8/255)")

print("\n Визуализация: 5 чистых + 5 PGD")
plot_group(images, adv_pgd, pred_clean, pred_pgd, conf_clean, conf_pgd, labels, "PGD (ε=8/255)")

print("\n Визуализация: 5 чистых + 5 DeepFool")
plot_group(images, adv_df, pred_clean, pred_df, conf_clean, conf_df, labels, "DeepFool")


# 9. ТАБЛИЦА

print("\n" + "="*65)
print("Точность модели на CIFAR-10 (оценка на 500 изображениях)")
print("="*65)
print(f"{'Метод':<15} | {'Точность (%)':<12} | {'Падение (%)':<10}")
print("-"*65)
print(f"{'Чистые данные':<15} | {acc_clean_500:>12.2f} | {'—':<10}")
print(f"{'FGSM':<15} | {acc_fgsm_500:>12.2f} | {acc_clean_500 - acc_fgsm_500:>9.2f}")
print(f"{'PGD-20':<15} | {acc_pgd_500:>12.2f} | {acc_clean_500 - acc_pgd_500:>9.2f}")
print(f"{'DeepFool':<15} | {acc_df_500:>12.2f} | {acc_clean_500 - acc_df_500:>9.2f}")
print("="*65)

