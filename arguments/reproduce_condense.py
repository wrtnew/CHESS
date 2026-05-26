def set_arguments(args):
    """Specific arguments for reproduce our condensed data
       The metric choice does not matter much.
       But, you should adjust lr_img according to the metric.
    """
    if args.dataset == 'imagenet':
        args.data_name = "{}{}".format(args.dataset, args.nclass)
    else:
        args.data_name = "{}".format(args.dataset)#崔老师是5200-n—retrain，别的就叫数据集-cnn
    args.pretrain_dir = "/home/wrt/pretrain"
    if args.dataset != 'imagenet':#崔老师是wificnn，别的就叫数据集-cnn
        args.net_type = 'wifi_cnn'
        args.depth = 3
        args.niter = 50000
        args.lr_img = 1e-6#调这个，崔老师数据集是1e-6，ntu是1e-5，pamap2是1e-5,uci是1e-4
        args.lr= 1e-3#除了ntu是5e-4，别的都是1e-3
        # args.train_lr= 1e-1
        # args.lr_img = 1e3 * args.ipc
        args.pretrain_amount = 30
        args.pretrain_epochs = 200
        if args.dataset == 'wifi':
            args.data_dir = '<>'
        else:
            raise AssertionError("Not supported dataset!")
    else:
        args.imagenet_dir = '<>'
        args.net_type = 'resnet_ap'
        args.depth = 5
        args.niter = 500

        if args.factor >= 3 and args.ipc >= 20:
            args.decode_type = 'bound'


    args.exp_name = 'Ipc{}_Fac{}_Lr{}_Npm{}_Ic{}_Bsr{}_Bss{}'.format(args.ipc, 
                                                                args.factor, 
                                                                args.lr_img, 
                                                                args.num_premodel, 
                                                                args.iter_calib,
                                                                args.batch_real,
                                                                args.batch_syn_max)
    # Result folder name
    if args.test:
        args.save_dir = './test_results/'
    else:
        args.save_dir = "./your_saved_results/"
    args.batch_syn_max=64
    args.batch_real=64
    args.load_memory=False
    args.rank = 5  # 秩 (Rank)
    args.seglen = 100  # 切比雪夫分段长度
    args.degree = 2

    return args
