import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import numpy as np
import random
import torch.nn.functional as F
import os
import platform


def pgd_attack(model, images, labels, epsilon=8 / 255, alpha=2 / 255, num_iter=7):

    model.eval()
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1)

    for _ in range(num_iter):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.data

        adv_images = adv_images + alpha * grad.sign()
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        adv_images = (images + eta).detach()

    model.train()
    return torch.clamp(adv_images, 0, 1).detach()


def evaluate_robustness(model, test_loader, device, attack_fn=None, **attack_kwargs):

    model.eval()
    total = correct = 0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            if attack_fn is not None:
                images = attack_fn(model, images, labels, **attack_kwargs)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % 20 == 0:
                print(f"  → {batch_idx + 1}/{len(test_loader)}", end='\r')
    print()
    acc = 100.0 * correct / total
    return acc


def adversarial_training_pgd(model, train_loader, test_loader, device, epochs=12):

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_test_acc = 0.0
    best_epoch = 0


    print(" НАЧАЛО СОСТЯЗАТЕЛЬНОГО ОБУЧЕНИЯ (PGD)")
    print(f"Эпох: {epochs} | Batch size: 128 | ε=8/255 | α=2/255 | iters=7")


    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            adv_images = pgd_attack(model, images, labels, epsilon=8 / 255, alpha=2 / 255, num_iter=7)

            all_images = torch.cat([images, adv_images], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad()
            outputs = model(all_images)
            loss = criterion(outputs, all_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += all_labels.size(0)
            train_correct += predicted.eq(all_labels).sum().item()

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"  Батч {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100. * train_correct / train_total:.2f}%")

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        print(f"\n[Эпоха {epoch + 1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        test_acc = evaluate_robustness(model, test_loader, device, attack_fn=None)
        print(f"  Test Acc (чистые): {test_acc:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'resnet20_adv_trained_pgd.pth')
            print(f"   Сохранена лучшая модель (эпоха {best_epoch}, точность {best_test_acc:.2f}%)")

        scheduler.step()


    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"Лучшая точность на тесте: {best_test_acc:.2f}% (эпоха {best_epoch})")
    print(f"Модель сохранена в 'resnet20_adv_trained_pgd.pth'")


    return model, best_test_acc


if __name__ == '__main__':

    # 1. Автоматическое определение устройства и настройка загрузчика


    print(" ЗАПУСК СОСТЯЗАТЕЛЬНОГО ОБУЧЕНИЯ (PGD)")


    # Определяем устройство
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f" Используется GPU: {torch.cuda.get_device_name(0)}")
        num_workers = 4
        pin_memory = True
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f" Используется Apple MPS (Metal Performance Shaders)")
        num_workers = 0  # MPS не поддерживает многопроцессную загрузку
        pin_memory = False
    else:
        device = torch.device("cpu")
        print(f" GPU/MPS недоступен, используется CPU")
        num_workers = 0
        pin_memory = False

    print(f"  Устройство: {device}")
    print(f" num_workers: {num_workers} | pin_memory: {pin_memory}")

    # Трансформации
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Датасеты с безопасными параметрами для всех платформ
    trainset = CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    testset = CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    trainloader = DataLoader(
        trainset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False if num_workers == 0 else True
    )
    testloader = DataLoader(
        testset,
        batch_size=100,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=False if num_workers == 0 else True
    )

    print(f" Обучающая выборка: {len(trainset)} изображений")
    print(f" Тестовая выборка: {len(testset)} изображений")


    # 2. Загрузка предобученной модели

    print("\n Загрузка предобученной ResNet-20...")
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet20',
        pretrained=True,
        trust_repo=True
    )
    model = model.to(device)
    model.train()
    print(" Модель загружена и переведена в режим обучения.")


    # 3. Дообучение модели (12 эпох)

    print("\n Начало дообучения с PGD-атакой...")
    model_adv, best_acc = adversarial_training_pgd(model, trainloader, testloader, device, epochs=12)




