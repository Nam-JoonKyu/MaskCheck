import torch
import torch.nn.functional as F
import torch.optim as optim
import argparse
from model import MaskNet
from dataloader import MaskDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

preprocess = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main(args):
    num_channels = [3, 64, 128, 256, 512, 512, 512]
    model = MaskNet(num_channels, separable=args.separable)
    model.to(device)

    train_ds = MaskDataset(root='dataset', split='train', preprocess=preprocess)
    test_ds = MaskDataset(root='dataset', split='test', preprocess=preprocess, no_aug=True)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_ds, batch_size=args.batch_size,
                             num_workers=args.num_workers)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


    train_losses = []
    test_losses = []
    acc_list = []
    best_acc = -1

    for epoch in range(args.epochs):
        train_loss = []
        test_loss = []
        acc = []
        model.train()
        pbar = tqdm(train_loader)
        for images, y in pbar:
            optimizer.zero_grad()
            N = images[0].size(0)
            y = y.to(device)
            ori_aug1_aug2 = torch.cat(images, dim=0).to(device)

            logits = model(ori_aug1_aug2)
            logits_o, logits_1, logits_2 = torch.split(logits, N)
            ori_loss = F.cross_entropy(logits_o, y)
            jsd = train_ds.aug.jensen_shannon(logits_o, logits_1, logits_2)
            loss = ori_loss + 12 * jsd

            loss.backward()
            optimizer.step()
            train_loss.append(ori_loss.item())
            pbar.set_description(f'E:{epoch + 1:3d}|L:{ori_loss:.4f}, {jsd:.4f}|lr:{scheduler.get_last_lr()[0]:.2e}')

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)


        scheduler.step()

        if not args.fast:
            model.eval()
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred = model(x)
                    loss = F.cross_entropy(pred, y)
                    pred = torch.argmax(pred, dim=-1)

                    acc_batch = (pred==y).float().mean()

                    test_loss.append(loss.item())
                    acc.append(acc_batch.item())

                test_loss = sum(test_loss) / len(test_loss)
                acc = sum(acc) / len(acc)

                print(f'Acc:{acc*100:.2f} VL:{test_loss:.4f}')

                test_losses.append(test_loss)
                acc_list.append(acc)

            if best_acc < acc:
                best_acc = acc

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, './checkpoint/masknet.ckpt')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'train_losses': train_losses,
                'test_losses': test_losses,
                'acc_list': acc_list,
                'best_acc': best_acc
            }, 'results.ckpt')

    torch.save({
        'train_losses': train_losses,
        'test_losses': test_losses,
        'acc_list': acc_list,
        'best_acc': best_acc
    }, 'results.ckpt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PointNet')
    parser.add_argument(
        '--epochs',
        type=int,
        default=200
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--separable',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    main(args)