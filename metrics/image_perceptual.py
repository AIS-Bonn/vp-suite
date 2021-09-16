import lpips

# APPLIES TO ALL METRICS:
# expected shape: [c, h, w]
# expected value range: [-1.0, 1.0]
lpips_alex = lpips.LPIPS(net='alex') # LPIPS ver. 0.1.4
def LPIPS(pred, target):
    '''
    input type: torch.tensor (torch.float)
    '''
    return lpips_alex(target, pred)