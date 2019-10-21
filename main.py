import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from config import config
from datasets import VOCDataset
from models import FCN
from utils import load_checkpoints


def train(models, writer, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VOCDataset(config.dataset_dir, year='2011', image_set='train', download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataset = VOCDataset(config.dataset_dir, year='2011', image_set='val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    train_iter = iter(train_loader)
    for idx in range(config.load_iter, config.train_iters):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
        images = images.to(device)
        labels = labels.to(device)

        logits = models(images)
        loss = models.criterion(logits, labels)

        models.optimizer.zero_grad()
        loss.backward()
        models.optimizer.step()

        if (idx + 1) % config.print_step == 0:
            print("Iteration: {}/{}, loss: {:.4f}".format(idx + 1, config.train_iters, loss.item()))

        if (idx + 1) % config.tensorboard_step == 0:
            writer.add_scalar('Loss/train', loss.item(), idx + 1)
            with torch.no_grad():
                images, labels = next(iter(val_loader))
                images = images.to(device)
                labels = labels.to(device)

                logits = models(images)
                loss = models.criterion(logits, labels)
                writer.add_scalar('Loss/val', loss.item(), idx + 1)

        if (idx + 1) % config.checkpoint_step == 0:
            checkpoint_path = os.path.join(config.checkpoint_dir, '{:06d}-{}.ckpt'.format(idx + 1, config.network))
            torch.save(models.network.state_dict(), checkpoint_path)


def test(models, device):
    pass


def main():
    if not os.path.exists(os.path.join(config.tensorboard_dir, config.network)):
        os.makedirs(os.path.join(config.tensorboard_dir, config.network))
    if not os.path.exists(config.checkpoint_dir):
        os.makedirs(config.checkpoint_dir)

    device = torch.device('cuda:0' if config.use_cuda else 'cpu')
    models = FCN(config).to(device)
    if config.load_iter != 0:
        load_checkpoints(models.network, config.network, config.checkpoint_dir, config.load_iter)

    if config.is_train:
        models.train()
        writer = SummaryWriter(log_dir=os.path.join(config.tensorboard_dir, config.network))
        train(models, writer, device)
    else:
        models.eval()
        test(models, device)


if __name__ == '__main__':
    main()
