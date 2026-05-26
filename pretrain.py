import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
import torch.nn as nn
from data import TensorDataset
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import torch.optim as optim
from train import define_model, train_epoch, validate
import torch.nn.functional as F
import random
from torchvision import transforms
def pretrain(args, logger, device='cuda'):

    x_train = np.load('/home/wrt/ActR/x_train.npy')
    y_train = np.load('/home/wrt/ActR/y_train.npy')
    x_test = np.load('/home/wrt/ActR/x_test.npy')
    y_test = np.load('/home/wrt/ActR/y_test.npy')

    # 为了使数据能让CNN处理转换为图格式数据
    print(x_train.shape)
    x_train = torch.tensor(x_train.reshape(len(x_train), 1, 2000, 30))
    y_train = torch.tensor(y_train)
    x_test = torch.tensor(x_test.reshape(len(x_test), 1, 2000, 30))
    y_test = torch.tensor(y_test)
    # print(x_train[0])


    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=True)

    nclass =7

    # _, aug_rand = diffaug(args)#不做可训练增广

    criterion = nn.CrossEntropyLoss()

    logger(f"Start training {args.pretrain_amount} models for {args.pretrain_epochs} epochs")
    for model_id in range(args.pretrain_amount):

        init_path = os.path.join(args.pretrain_dir, f'premodel{model_id}_init.pth.tar')
        trained_path = os.path.join(args.pretrain_dir, f'premodel{model_id}_trained.pth.tar')
        model = define_model(args, nclass).to(device).float()
        torch.save(model.state_dict(), init_path)

        model.train()
        optim_net = optim.Adam(model.parameters(),
                              args.lr,
                              )
        scheduler = optim.lr_scheduler.MultiStepLR(optim_net,
                                                   milestones=[2 * args.pretrain_epochs // 3,
                                                               5 * args.pretrain_epochs // 6],
                                                   gamma=0.2)

        for epoch in range(args.pretrain_epochs):
            top1, _, loss = train_epoch(args,
                                        train_loader,
                                        model,
                                        criterion,
                                        optim_net,
                                        aug=None,
                                        mixup=args.mixup)
            top1_val, _, _ = validate(test_loader, model, criterion)
            logger(
                "<Pretraining {:2d}-th model>...[Epoch {:2d}] Train acc: {:.1f} (loss: {:.3f}), Val acc: {:.1f}".format(
                    model_id,
                    epoch,
                    top1,
                    loss,
                    top1_val))
            scheduler.step()

        torch.save(model.state_dict(), trained_path)

def set_all_seeds(seed=0):
    # Python 随机种子
    random.seed(seed)

    # NumPy 随机种子
    np.random.seed(seed)

    # PyTorch 随机种子（CPU + CUDA）
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

    # 禁用 CUDA 非确定性算法（确保可复现性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置 Python 哈希种子（避免哈希随机化）
    os.environ["PYTHONHASHSEED"] = str(seed)


if __name__ == '__main__':
    from misc.utils import Logger
    from arguments.arg_pretrain import args
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = True
    if args.seed > 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    set_all_seeds(0)
    os.makedirs(args.pretrain_dir, exist_ok=True)

    logger = Logger(args.pretrain_dir)
    logger(f"Pretrain models save dir: {args.pretrain_dir}")
    logger(args)
    pretrain(args, logger)