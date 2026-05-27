import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
# import time
import numpy.polynomial.chebyshev as cheb

# from data import TensorDataset
from data import ClassDataLoader, ClassMemDataLoader, MultiEpochsDataLoader
from common import define_model

from misc import utils

import random

from torch.nn import functional as F

from fast_pytorch_kmeans import KMeans

import torch

from misc.utils import init_sample

import os


class TensorDatasets(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        # images: NxCxHxW tensor
        self.images = images.detach().cpu().float()
        self.targets = labels.detach().cpu()
        self.transform = transform

    def __getitem__(self, index):
        sample = self.images[index]
        if self.transform != None:
            sample = self.transform(sample)

        target = self.targets[index]
        return sample, target

    def __len__(self):
        return self.images.shape[0]





def dist(x, y, method='mse'):
    """Distance objectives
    """
    if method == 'mse':
        dist_ = (x - y).pow(2).sum()
    elif method == 'l1':
        dist_ = (x - y).abs().sum()
    elif method == 'l1_mean':
        n_b = x.shape[0]
        dist_ = (x - y).abs().reshape(n_b, -1).mean(-1).sum()
    elif method == 'cos':
        x = x.reshape(x.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        dist_ = torch.sum(1 - torch.sum(x * y, dim=-1) /
                          (torch.norm(x, dim=-1) * torch.norm(y, dim=-1) + 1e-6))

    return dist_


def add_loss(loss_sum, loss):
    if loss_sum == None:
        return loss
    else:
        return loss_sum + loss





import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# 引入所有可能的多项式库
import numpy.polynomial.chebyshev as cheb
import numpy.polynomial.legendre as leg
import numpy.polynomial.hermite as herm
import numpy.polynomial.polynomial as poly


class  ChessSynthesizer():
    """
    Modeling: Data = (Temporal_Basis * Coeffs) * Sigma * Spatial_Basis^T
    """

    def __init__(self, args, nclass, nchannel, hs, ws, device='cuda'):
        self.ipc = args.ipc
        self.nclass = nclass
        self.nchannel = nchannel  # 1
        self.time_steps = hs  # 2000
        self.subcarriers = ws  # 30
        self.device = device
        self.model = None

        # --- 超参数设置 ---
        self.rank = args.rank
        self.seg_len = args.seglen
        self.degree = args.degree

        # [新增] 获取基底类型，默认为 chebyshev
        self.basis_type = getattr(args, 'basis', 'chebyshev').lower()

        # 校验
        if self.time_steps % self.seg_len != 0:
            raise ValueError(f"Time steps {self.time_steps} must be divisible by seg_len {self.seg_len}")
        self.num_segments = self.time_steps // self.seg_len
        self.num_coeffs = self.degree + 1

        print(f"\n[Decomposed Synthesizer] Rank: {self.rank}, Segs: {self.num_segments}, Degree: {self.degree}")
        print(f"[Basis Type] Using: {self.basis_type}")

        # --- 1. 预计算时间维基底矩阵 (固定) ---
        # Basis shape: (num_coeffs, seg_len) -> e.g. (11, 200)
        basis_numpy = self._get_basis_matrix(self.basis_type, self.seg_len, self.degree)

        self.temporal_basis = torch.from_numpy(basis_numpy).float().to(self.device)

        # 预计算伪逆矩阵，用于初始化时的投影: Coeffs = U * Basis_pinv
        # shape: (seg_len, num_coeffs) -> (200, 11)
        basis_pinv = np.linalg.pinv(basis_numpy)
        self.temporal_basis_pinv = torch.from_numpy(basis_pinv).float().to(self.device)

        # --- 2. 定义可学习参数 ---
        num_syn_data = self.nclass * self.ipc

        # A. 时间维系数 (通用命名)
        # 初始化为一个很小的值，具体值会在 init() 中被覆盖
        self.temporal_coeffs = nn.Parameter(torch.randn(
            num_syn_data, self.rank, self.num_segments, self.num_coeffs, device=self.device
        ) * 0.01)

        # B. 奇异值
        self.sigma = nn.Parameter(torch.rand(num_syn_data, self.rank, device=self.device))

        # C. 空间/信道基
        self.v_channel = nn.Parameter(torch.randn(
            num_syn_data, self.subcarriers, self.rank, device=self.device
        ) * 0.01)

        # 标签
        self.targets = torch.tensor([np.ones(self.ipc) * i for i in range(nclass)],
                                    dtype=torch.int64,
                                    requires_grad=False,
                                    device=self.device).view(-1)

        self.factor = max(1, args.factor)
        self.decode_type = args.decode_type

        # 模型加载逻辑 (保持不变)
        from common import define_model
        self.model = define_model(args, nclass).to(device)
        if args.num_premodel > 0:
            import random, os
            slkt_model_id = random.randint(0, args.num_premodel - 1)
            final_path = os.path.join(args.pretrain_dir, 'premodel{}_trained.pth.tar'.format(slkt_model_id))
            # 兼容性处理
            if os.path.exists(final_path):
                self.model.load_state_dict(torch.load(final_path))
                print(f"Loaded pre-trained model: {final_path}")
            self.model.eval()

        print(f"Total Params: {self.temporal_coeffs.numel() + self.sigma.numel() + self.v_channel.numel()}")

    def _get_basis_matrix(self, basis_type, length, degree):
        """根据类型生成基底矩阵，返回 shape: (degree+1, length)"""
        # 定义域 [-1, 1] 通常是大多数正交多项式的最佳区间
        x = np.linspace(-1, 1, length)

        if basis_type == 'chebyshev':
            # 切比雪夫 (最佳边缘拟合)
            return cheb.chebvander(x, degree).T

        else:
            raise ValueError(f"Unsupported basis type: {basis_type}")

    def reconstruct_data(self):
        """ X = U * S * V^T """
        # 1. U (Time): Coeffs @ Basis
        # (N, Rank, Segs, Coeffs) @ (Coeffs, SegLen) -> (N, Rank, Segs, SegLen)
        u_segments = torch.matmul(self.temporal_coeffs, self.temporal_basis)

        # Flatten segments: (N, Rank, Time)
        u_time = u_segments.view(u_segments.shape[0], self.rank, -1)

        # 2. S (Sigma)
        s_diag = self.sigma.unsqueeze(-1)  # (N, Rank, 1)

        # 3. V (Space)
        v_t = self.v_channel.permute(0, 2, 1)  # (N, Rank, 30)

        # 4. Combine: (U * S) @ V^T
        weighted_u = (u_time * s_diag).permute(0, 2, 1)  # (N, 2000, Rank)
        reconstructed = torch.matmul(weighted_u, v_t)  # (N, 2000, 30)

        return reconstructed.unsqueeze(1)  # (N, 1, 2000, 30)

    @property
    def data(self):
        return self.reconstruct_data()

    def parameters(self):
        return [self.temporal_coeffs, self.sigma, self.v_channel]

    def init(self, dataset, loader, init_type='dream'):

        print(f"Initializing with strategy: {init_type} (Basis: {self.basis_type})...")

        all_selected_images = []

        # 遍历类别获取真实数据
        for c in range(self.nclass):
            if hasattr(loader, 'class_sample'):
                n_total_c = len(loader.class_indices[c]) if hasattr(loader, 'class_indices') else 2000
                img_real, _ = loader.class_sample(c, n_total_c)
            else:
                # 简单 Fallback，如果不使用 class_sample
                indices = [i for i, label in enumerate(dataset.targets) if label == c]
                # 随机采样一些
                sel_indices = np.random.choice(indices, min(len(indices), 1024), replace=False)
                img_real = torch.stack([dataset[i][0] for i in sel_indices]).to(self.device)
            self.model.eval()
            select = init_sample(img_real, self.model)
            query_idxs = select.query_no_pca(self.ipc)  # 获取 IPC 个最佳样本
            sel_img = img_real[query_idxs].detach()
            all_selected_images.append(sel_img)



        # 拼接
        selected_tensor = torch.cat(all_selected_images, dim=0).to(self.device)
        if selected_tensor.dim() == 4:
            selected_tensor = selected_tensor.squeeze(1)  # (N, 2000, 30)

        # 执行分解 (SVD + Temporal Basis Projection)
        with torch.no_grad():
            # A. SVD 分解
            U, S, Vh = torch.linalg.svd(selected_tensor, full_matrices=False)

            # B. 截断保留前 Rank 个分量
            U_k = U[:, :, :self.rank]  # (N, 2000, R)
            S_k = S[:, :self.rank]  # (N, R)
            Vh_k = Vh[:, :self.rank, :]  # (N, R, 30)

            # C. 赋值 Sigma
            self.sigma.data.copy_(S_k)

            # D. 赋值 V_channel (Spatial Basis)
            # Vh 是 V^T，我们要存储 V (N, 30, R)，即 Vh 的转置
            self.v_channel.data.copy_(Vh_k.permute(0, 2, 1))

            # E. 计算时间维多项式系数 (Temporal Basis Coeffs)
            # U_k shape: (N, 2000, R)
            # 1. 重排为分段格式: (N, R, Num_Segs, Seg_Len)
            u_reshaped = U_k.permute(0, 2, 1).contiguous().view(
                self.nclass * self.ipc, self.rank, self.num_segments, self.seg_len
            )

            # 2. 投影求解系数: Coeffs = Data @ Basis_PseudoInverse
            # Data: (..., SegLen)
            # Basis_pinv: (SegLen, NumCoeffs)
            # Result: (..., NumCoeffs)
            coeffs_init = torch.matmul(u_reshaped, self.temporal_basis_pinv)

            # F. 赋值 Coeffs
            self.temporal_coeffs.data.copy_(coeffs_init)

        print(f"Initialization complete using {self.basis_type} basis.")

    def sample(self, c, max_size=128):
        """Sample synthetic data per class"""
        idx_from = self.ipc * c
        idx_to = self.ipc * (c + 1)

        coeffs = self.temporal_coeffs[idx_from:idx_to]
        sigma = self.sigma[idx_from:idx_to]
        v = self.v_channel[idx_from:idx_to]

        # 局部重构
        u_segments = torch.matmul(coeffs, self.temporal_basis)
        u_time = u_segments.view(coeffs.shape[0], self.rank, -1)

        s_diag = sigma.unsqueeze(-1)
        weighted_u = (u_time * s_diag).permute(0, 2, 1)
        v_t = v.permute(0, 2, 1)

        reconstructed = torch.matmul(weighted_u, v_t).unsqueeze(1)

        return reconstructed, self.targets[idx_from:idx_to]

    # loader 和 test 函数保持不变 ...
    def loader(self, args, augment=True):
        full_data = self.reconstruct_data().detach().cpu()
        full_targets = self.targets.detach().cpu()
        train_dataset = TensorDataset(full_data, full_targets)
        nw = 0 if not augment else args.workers
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=nw)
        return train_loader

    def test(self, args, val_loader, logger, bench=True):
        loader = self.loader(args, args.augment)
        from test import test_data
        convnet_result = test_data(args, loader, val_loader, test_resnet=False, logger=logger)
        return convnet_result


def innerloss(img_real, img_syn, model):
    """Matching losses: feature mean + (optional) std matching.

    Std matching is only enabled when both batches have >=2 samples,
    otherwise unbiased std produces NaN (division by n-1=0).
    """
    with torch.no_grad():
        _, fea_1_tg, fea_2_tg, fea_3_tg = model(img_real, return_features=True)
    _, fea_1, fea_2, fea_3 = model(img_syn, return_features=True)

    # --- Mean matching (always on) ---
    loss_1 = torch.mean((fea_1.mean(dim=0, keepdim=True) - fea_1_tg.mean(dim=0, keepdim=True)) ** 2)
    loss_2 = torch.mean((fea_2.mean(dim=0, keepdim=True) - fea_2_tg.mean(dim=0, keepdim=True)) ** 2)
    loss_3 = torch.mean((fea_3.mean(dim=0, keepdim=True) - fea_3_tg.mean(dim=0, keepdim=True)) ** 2)
    loss_mse = loss_1 + loss_2 + loss_3

    loss = loss_mse

    if fea_3.shape[0] >= 2 and fea_3_tg.shape[0] >= 2:
        loss_1_var = torch.mean((fea_1.std(dim=0, unbiased=True) - fea_1_tg.std(dim=0, unbiased=True)) ** 2)
        loss_2_var = torch.mean((fea_2.std(dim=0, unbiased=True) - fea_2_tg.std(dim=0, unbiased=True)) ** 2)
        loss_3_var = torch.mean((fea_3.std(dim=0, unbiased=True) - fea_3_tg.std(dim=0, unbiased=True)) ** 2)
        loss_var = loss_1_var + loss_2_var + loss_3_var
        loss = loss + 0.1*loss_var

    loss = loss * fea_3.numel()
    return loss





def interloss(img_syn, label_syn, trained_model):
    logits = trained_model(img_syn, return_features=False)
    loss = F.cross_entropy(logits, label_syn)

    return loss


def z_score_normalize(train_data, test_data):
    mean = train_data.mean()
    std = train_data.std()
    train_norm = (train_data - mean) / std
    test_norm = (test_data - mean) / std
    return train_norm, test_norm


def condense(args, logger, device='cuda'):
    """Optimize condensed data
    """
    wandb.init(
        project="1-continue",
        config=vars(args),
        notes=f"{args.dataset} IPC{args.ipc}",
        tags=["DANCE", args.dataset]
    )
    # 记录数据集元数据
    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])


    x_train = np.load('/home/wrt/ActR/x_train.npy')
    y_train = np.load('/home/wrt/ActR/y_train.npy')
    x_test = np.load('/home/wrt/ActR/x_test.npy')
    y_test = np.load('/home/wrt/ActR/y_test.npy')
    # 为了使数据能让CNN处理转换为图格式数据
    print(x_train.shape)
    x_train = torch.tensor(x_train.reshape(len(x_train), 1, 2000, 30))
    y_train = torch.tensor(y_train,dtype=int)
    x_test = torch.tensor(x_test.reshape(len(x_test), 1, 2000, 30))
    y_test = torch.tensor(y_test,dtype=int)



    print(x_train.shape)

    train_dataset = TensorDatasets(x_train, y_train)
    test_dataset = TensorDatasets(x_test, y_test)
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=True)

    print("Datasets created.")
    if args.load_memory:
        loader_real = ClassMemDataLoader(train_dataset, batch_size=args.batch_real)
    else:
        loader_real = ClassDataLoader(train_dataset,
                                      batch_size=args.batch_real,
                                      num_workers=args.workers,
                                      shuffle=True,
                                      pin_memory=True,
                                      drop_last=True)
    print('1')
    nclass = y_test.max() + 1
    nch, hs, ws = train_dataset[0][0].shape
    print(nch, hs, ws)

    # Define syn dataset
    synset = ChessSynthesizer(args, nclass, nch, hs, ws)
    synset.init(train_dataset, loader_real, init_type='random')
    print("\n" + "=" * 40)
    print("TRAINABLE PARAMETERS DETAILED REPORT")
    print("=" * 40)

    total_params = 0
    param_names = ['cheb_coeffs (Time)', 'sigma (Weights)', 'v_channel (Space)']

    # synset.parameters() 返回的是列表 [cheb_coeffs, sigma, v_channel]
    for name, param in zip(param_names, synset.parameters()):
        num_params = param.numel()
        print(f"{name:<20} | Shape: {str(list(param.shape)):<25} | Count: {num_params}")
        total_params += num_params

    print("-" * 40)
    print(f"Total Trainable Params : {total_params}")

    # 计算原始数据大小作为对比
    raw_size = (args.ipc * nclass) * hs * ws
    print(f"Equivalent Raw Data    : {raw_size}")
    print(f"Compression Ratio      : {raw_size / total_params:.2f}x")
    print("=" * 40 + "\n")

    # Define augmentation function
    # aug, _ = diffaug(args)


    # Data distillation
    # optim_img = torch.optim.SGD(synset.parameters(), lr=args.lr_img, momentum=args.mom_img)
    optim_img = torch.optim.Adam(synset.parameters(), lr=args.lr_img)
    ts = utils.TimeStamp(args.time)

    n_iter = args.niter
    # n_iter = 2
    it_log = 10

    it_test = np.arange(0, n_iter + 1, 200).tolist()
    # it_test = np.arange(0, n_iter + 1, 1).tolist()

    # it_test = [n_iter // 10, n_iter // 5, n_iter // 2, n_iter]

    logger(f"\n CHESS: Start condensing for {n_iter} iteration")

    best_convnet = -1
    best_resnet = -1
    model_init = define_model(args, nclass).to(device)
    model_final = define_model(args, nclass).to(device)
    model_interval = define_model(args, nclass).to(device)
    # start_time = time.time()
    for it in range(n_iter):
        if args.num_premodel > 0:
            slkt_model_id = random.randint(0, args.num_premodel - 1)
            init_path = os.path.join(args.pretrain_dir, 'premodel{}_init.pth.tar'.format(slkt_model_id))
            final_path = os.path.join(args.pretrain_dir, 'premodel{}_trained.pth.tar'.format(slkt_model_id))
            model_init.load_state_dict(torch.load(init_path))
            model_final.load_state_dict(torch.load(final_path))

            l = torch.rand(1).cuda()
            for param_C, param_A, param_B in zip(model_interval.parameters(), model_init.parameters(),
                                                 model_final.parameters()):
                param_C.data.copy_(l * param_A.data + (1 - l) * param_B.data)
        else:
            slkt_model_id = random.randint(0, 4)
            final_path = os.path.join(args.pretrain_dir, 'premodel{}_trained.pth.tar'.format(slkt_model_id))
            model_final.load_state_dict(torch.load(final_path))

        '''detach the model'''
        # for name, param in model.named_parameters():
        #     param = param.detach()

        inne_loss = 0
        fishe_loss = 0
        dop_loss = 0
        div_loss_meter = 0
        # synset.data.data = torch.clamp(synset.data.data, min=0., max=1.)

        ts.set()

        # Update synset (inner-class view)
        for c in range(nclass):
            img_syn, lab_syn = synset.sample(c, max_size=args.batch_syn_max)
            img, lab = loader_real.class_sample(c)

            ts.stamp("data")

            ts.stamp("aug")

            inner_loss = innerloss(img, img_syn, model_interval)
            total_loss = inner_loss

            inne_loss += inner_loss.item()


            ts.stamp("loss")

            optim_img.zero_grad()
            # total_loss_1.backward()
            total_loss.backward()


            optim_img.step()
            ts.stamp("backward")

        ts.flush()

        # Update syn set (inter-class view)
        calib_loss_total = 0
        fisher_regular = 0
        if args.iter_calib > 0:
            for _ in range(args.iter_calib):
                for c in range(nclass):
                    img_syn, label_syn = synset.sample(c, max_size=args.batch_syn_max)
                    img_aug = img_syn
                    loss = interloss(img_aug, label_syn, model_final)
                    loss_inter = loss
                    calib_loss_total += loss.item()
                    optim_img.zero_grad()
                    loss_inter.backward()
                    optim_img.step()


        else:
            pass

            # 重置计时器，准备记录下一个 100 步
            # start_time = time.time()
        # Logging
        # if it % it_log == 0:
        #     logger(
        #         f"{utils.get_time()} (Iter {it:3d}) inter-loss: {calib_loss_total/nclass/args.iter_calib:.8f}  inner-loss: {inne_loss/nclass:.8f}  fish-loss: {fishe_loss / nclass:.8f} fisher regu: {fisher_regular/nclass:.8f}")
        if it % it_log == 0:
            logger(
                f"{utils.get_time()} (Iter {it:3d}) inter-loss: {calib_loss_total / nclass / args.iter_calib:.8f}  inner-loss: {inner_loss / nclass:.8f}  dop_loss: {dop_loss / nclass:.8f} ")
        wandb.log({"fisher-loss": fishe_loss / nclass})
        wandb.log({"inter-loss": calib_loss_total / nclass / args.iter_calib})
        wandb.log({"inner-loss": inne_loss / nclass})
        wandb.log({"div-loss": div_loss_meter / nclass})
        save_best = 0
        if (it + 1) in it_test:
            conv_result = synset.test(args, test_loader, logger)
            wandb.log({"current result": conv_result})

            if conv_result > best_convnet:
                best_convnet = conv_result
                save_best = 1

                logger("->->->->->->->->->->->->-> Best Result: {:.1f}".format(best_convnet))
                wandb.log({"bes test result": best_convnet})

            if not args.test:
                # save_img(os.path.join(args.save_dir, f'img{it+1}.png'),
                #      synset.data,
                #      unnormalize=False,
                #      dataname=args.dataset)
                torch.save(
                    [synset.data.detach().cpu(), synset.targets.cpu()],
                    os.path.join(args.save_dir, 'data_{}.pt'.format(it + 1)))
                logger("img and data saved!")

                if save_best:
                    CUSTOM_SAVE_PATH = '/home/wrt/chebyshev/dontuse/chess/'

                    # 自动创建目录
                    if not os.path.exists(CUSTOM_SAVE_PATH):
                        os.makedirs(CUSTOM_SAVE_PATH)

                    print(f"Saving numpy files to independent path: {CUSTOM_SAVE_PATH}")

                    # --- 保存 x_train.npy & y_train.npy ---
                    # 获取合成数据，转为numpy
                    syn_data_np = synset.data.detach().cpu().numpy()
                    syn_targets_np = synset.targets.detach().cpu().numpy()

                    # 维度压缩: (N, 1, T, F) -> (N, T, F)
                    if syn_data_np.ndim == 4 and syn_data_np.shape[1] == 1:
                        syn_data_np = syn_data_np.squeeze(1)

                    np.save(os.path.join(CUSTOM_SAVE_PATH, 'x_train.npy'), syn_data_np)
                    np.save(os.path.join(CUSTOM_SAVE_PATH, 'y_train.npy'), syn_targets_np)

                    # # --- 保存 x_test.npy & y_test.npy ---
                    # # x_test 和 y_test 是函数开头加载的变量
                    # x_test_np = x_test.cpu().numpy() if isinstance(x_test, torch.Tensor) else x_test
                    # y_test_np = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
                    #
                    # if x_test_np.ndim == 4 and x_test_np.shape[1] == 1:
                    #     x_test_np = x_test_np.squeeze(1)
                    #
                    # np.save(os.path.join(CUSTOM_SAVE_PATH, 'x_test.npy'), x_test_np)
                    # np.save(os.path.join(CUSTOM_SAVE_PATH, 'y_test.npy'), y_test_np)

                    # =======================================================

                    logger("best data saved to custom path")
    wandb.finish()


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


# 调用函数，设置所有种子为 0


if __name__ == '__main__':
    from misc.utils import Logger
    from arguments.arg_condense import args
    import torch.backends.cudnn as cudnn
    import json

    assert args.ipc > 0

    cudnn.benchmark = True
    set_all_seeds(0)

    if not args.test:
        os.makedirs(args.save_dir, exist_ok=True)

        logger = Logger(args.save_dir)
        logger(f"Save dir: {args.save_dir}")

        with open(os.path.join(args.save_dir, 'args.log'), 'w') as f:
            json.dump(args.__dict__, f, indent=3)
    else:
        logger = print

    condense(args, logger)
