def set_arguments(args):
    """Specific arguments for reproduce our condensed data
       The metric choice does not matter much.
       But, you should adjust lr_img according to the metric.
    """
    if args.dataset == 'imagenet':
        args.data_name = "{}{}".format(args.dataset, args.nclass)
    else:
        args.data_name = "{}".format(args.dataset)
    args.pretrain_dir = "/home/wrt/pretrain"
    if args.dataset != 'imagenet':
        # args.net_type = 'seedth_cnn'
        # args.net_type = 'ecg_cnn'
        # args.net_type = 'ut_har_cnn'
        # args.net_type = 'pafa_cnn'
        # args.net_type = 'sam'
        # args.net_type = 'sam_1000_cnn'
        # args.net_type = 'deap_cnn'
        # args.net_type = 'sle_cnn'
        # args.net_type = 'cr_res'
        # args.net_type ='cr_den'
        # args.net_type ='cr_eff'
        # args.net_type ='cr_in'
        # args.net_type ='cr_shu'
        # args.net_type = 'ts_cnn'
        args.depth = 3
        args.niter = 10000
        args.pretrain_amount = 20
        args.pretrain_epochs = 100
        if args.dataset[:5] == 'cifar':
            args.data_dir = '<>'
        elif args.dataset == 'svhn':
            args.data_dir = '<>'
            # if args.factor == 1 and args.ipc == 1:
            #     # In this case, evaluation w/o mixup is much more effective
            #     args.mixup = 'vanilla'
            #     args.dsa_strategy = 'color_crop_cutout_scale_rotate'
        elif args.dataset == 'fashion':
            args.data_dir = '<>'
        elif args.dataset == 'mnist':
            args.data_dir = '<>'
        elif args.dataset == 'tinyimagenet':
            args.data_dir = '<>'
        elif args.dataset in ['imagenette', 'imagewoof', 'imagemeow', 'imagesquawk', 'imagefruit', 'imageyellow']:
            args.imagenet_dir = '<>'
            args.net_type = 'convnet'
            args.depth = 5
            args.niter = 10000
            args.pretrain_amount = 5
            args.pretrain_epochs = 80

            if args.nclass == 10:
                args.batch_real = 128
        else:
            raise AssertionError("Not supported dataset!")
    
    else:
        args.imagenet_dir = '<>'
        args.net_type = 'ntu_cnn'
        args.depth = 10
        args.niter = 10000

        # if args.ipc == 10:
        #     args.lr_img = 50.
        # elif args.ipc == 20:
        #     args.lr_img = 100.

        if args.nclass == 10:
            args.batch_real = 128
            
        # elif args.nclass == 100:
        #     args.lr_img = 1.


        if args.factor >= 3 and args.ipc >= 20:
            args.decode_type = 'bound'
    args.lr = 1e-3










    return args
