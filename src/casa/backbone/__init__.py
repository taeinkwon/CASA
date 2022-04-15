from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4
from .fcl import FCL
from .projection_head import ProjectionHead

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['backbone_type'] == 'FCL':
        return FCL(config['fcl'])

    elif config['backbone_type'] == 'PH':
        return ProjectionHead(config)
    
    else:
        raise ValueError(
            f"CASA.BACKBONE_TYPE {config['backbone_type']} not supported.")
