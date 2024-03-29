import argparse, torch, os, random
import numpy as np

def parse_args(run_type = 'terminal'):
    parser = argparse.ArgumentParser(description='Python Training')
    
    # Optimization options
    parser.add_argument('--network', default='resnet32', help='Network: resnet32')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batch-size', default=128, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--update-epoch', default=1, type=int, metavar='N', help='Update epoch')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_decay', default=0.01, type=float, help='learnign rate decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--wd', default=2e-4, type=float, help='weight decay factor for optimizer')
    parser.add_argument('--nesterov', action='store_true', help="Utilizing Nesterov")
    parser.add_argument('--scheduler', default='warmup', type=str, help='LR scheduler')#cosine  warmup
    parser.add_argument('--warmup', default=5, type=int, help='Warmup epochs')
        
    parser.add_argument('--aug_prob', default=0.5, type=float, help='Augmentation Coin-tossing Probability')
    parser.add_argument('--cutout', action='store_true', help='Utilizing Cutout')
    parser.add_argument('--cmo', action='store_true', help='Utilizing CMO')
    parser.add_argument('--posthoc_la', action='store_true', help='Posthoc LA for state update')
    parser.add_argument('--cuda', action='store_true', default=True,help='Use CUDA')
    parser.add_argument('--aug_type', default='autoaug_cifar')
    parser.add_argument('--sim_type', default='none')
    parser.add_argument('--max_d', type=int, default=30, help='max_d')

    parser.add_argument('--num_test', default=10, type=int, help='Curriculum Test')
    parser.add_argument('--accept_rate', type=float, default=0.6, help='Increasing accept ratio')
    parser.add_argument('--verbose', action='store_true', help='Debug on/off')
    parser.add_argument('--use_norm', action='store_true', help='Utilize Normed Linear')    

    parser.add_argument('--supper_classes',type = int ,default=10)
    # Checkpoints
    parser.add_argument('--out', default='/home/zz/wenhaibin/cifar/logs/results/', help='Directory to output the result')
    parser.add_argument('--data_dir', default='~/dataset/')
    
    # Miscs
    parser.add_argument('--workers', type=int, default=4, help='# workers')
    parser.add_argument('--seed', type=str, default='None', help='manual seed')
    parser.add_argument('--gpu', default= '1', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    
    # Dataset options
    parser.add_argument('--dataset', default='cifar100', help='Dataset: cifar100')
    parser.add_argument('--num_max', type=int, default=500, help='Number of samples in the maximal class')
    parser.add_argument('--imb_ratio', type=int, default=100, help='Imbalance ratio for data')
    
    # Method options
    parser.add_argument('--num_experts', type=int, default=3, help='Number of experts for RIDE')
    parser.add_argument('--ride_distill', action='store_true', help='Use RIDEWithDistill Loss')

    #loss fun
    parser.add_argument('--ce', action='store_true',default = False)
    parser.add_argument('--bs', action='store_true',default = False)
    parser.add_argument('--bcl', action='store_true',default = False)
    parser.add_argument('--ce_drw', action='store_true',default = False)
    parser.add_argument('--ldam_drw', action='store_true',default = False)
    parser.add_argument('--gml', action='store_true',default = False)
    parser.add_argument('--kps', action='store_true',default = False)
    parser.add_argument('--shike', action='store_true',default = False)
    
    # Mback
    parser.add_argument('--mgda',action='store_true',default=False)
    parser.add_argument('--chs',action='store_true',default=False)
    parser.add_argument('--pla',action='store_true',default=False)
    parser.add_argument('--pcg',action='store_true',default=False)
    parser.add_argument("--tasks",                         default=None)
    parser.add_argument("--mgda_mode",                     type = str,default='l2')#choices=['l2', 'loss', 'loss+',
    
    # cat out
    parser.add_argument('--out_cut',action='store_true',default=False)


    #LSF
    parser.add_argument('--MOOSF',action='store_true',default=False)


    # large
    parser.add_argument('--data_aug', default="vanilla", type=str, help='data augmentation type',
                    choices=('vanilla', 'CMO', 'CUDA', 'CUDA_CMO'))
    parser.add_argument('--use_randaug', action='store_true')
    parser.add_argument('--cuda_upgrade', default=1, type=int, help="upgrade DA level")
    parser.add_argument('--cuda_downgrade', default=1, type=int, help="downgrade DA level")
    
    if run_type == 'terminal':
        args = parser.parse_args()
        args.loss_fn,args.tasks =  lossfun(args)
        args.out = f'{args.out}{args.dataset}/{args.loss_fn}@N_{args.num_max}_ir_{args.imb_ratio}/'
    elif run_type =='jupyter':
        args = parser.parse_args(args=[])
 
    if args.gpu:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if args.MOOSF:
        args.pcg = True
        args.pla = True
        args.out_cut = True


    return args

def lossfun(args):
    tasks = []
    if args.ce:
        tasks.append('ce')
    if args.bs:
        tasks.append('bs')
    if args.kps:
        tasks.append('kps')
    if args.bcl:
        tasks.append('bcl')
        args.cmo = False
    if args.ce_drw:
        tasks.append('ce_drw')
    if args.ldam_drw:
        tasks.append('ldam_drw')
    if args.gml:
        tasks.append('gml')
        args.cmo = False
    if args.shike:
        tasks.append('shike')
        args.cmo = False
    #tasks = tasks[::-1]
    return  '-'.join(tasks),tasks 

def reproducibility(seed):
    if seed == 'None':
        return
    else:
        seed = int(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(seed)
        random.seed(seed)

def dataset_argument(args):
    if args.dataset == 'cifar100':
        args.network = 'resnet32'
        args.num_class = 100
    elif args.dataset == 'imgnet':
        args.num_class = 1000
        args.network = 'resnet50'
    elif args.dataset == 'inat18':
        args.num_class = 8142
        args.network = 'resnet50'
    else :
        args.num_class = 10
    args.out = f'{args.out}{args.dataset}/{args.loss_fn}@N_{args.num_max}_ir_{args.imb_ratio}/'
    return args
