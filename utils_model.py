import torch
import models


def fetch_model(pretrained_path, args, return_feat_maps=False):
    """Create model and initialize with weights

    args:
        pretrained_path ([type]): [description]

    Returns:
        torch.model: flownet model with pretrained weights
    """
    if args.flownet == 'SpyNet':
        flow_net = getattr(models, args.flownet)(nlevels=6, pretrained=True)
    elif args.flownet == 'Back2Future':
        flow_net = getattr(models, args.flownet)(
            pretrained=pretrained_path/'b2f_rm_hard.pth.tar')
    elif args.flownet == 'PWCNet':
        flow_net = models.pwc_dc_net(pretrained_path/'pwc_net_chairs.pth.tar')
    else:
        if 'BN' in args.flownet:
            flow_net_name = args.flownet[:args.flownet.rfind('_BN')]
            flow_net = getattr(models, flow_net_name)(batchNorm=True, return_feat_maps=return_feat_maps)
        else:
            flow_net = getattr(models, args.flownet)(return_feat_maps=return_feat_maps)

    if args.flownet in ['SpyNet', 'Back2Future', 'PWCNet']:
        print("=> using pre-trained weights for " + args.flownet)
    elif args.flownet in ['FlowNetC']:
        print("=> using pre-trained weights for FlowNetC")
        weights = torch.load(pretrained_path/'FlowNet2-C_checkpoint.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    elif args.flownet in ['FlowNetS']:
        print("=> using pre-trained weights for FlowNetS")
        weights = torch.load(pretrained_path/'flownets.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    elif args.flownet in ['FlowNetS_BN']:
        print("=> using pre-trained weights for FlowNetS_BN")
        weights = torch.load(pretrained_path/'flownets_bn.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    elif args.flownet in ['FlowNet2']:
        print("=> using pre-trained weights for FlowNet2")
        weights = torch.load(pretrained_path/'FlowNet2_checkpoint.pth.tar')
        flow_net.load_state_dict(weights['state_dict'])
    else:
        flow_net.init_weights()

    return flow_net
