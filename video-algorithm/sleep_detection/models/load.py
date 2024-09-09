from pathlib import Path

import torch

from torch import nn
from models.yolo import Detect, Model
from ai_common import setting


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

# torch 1.6.0


def attempt_load(weights, map_location=None, inplace=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(str(Path(str(w).strip().replace(
            "'", ''))), map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema')
                     else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        # logger.info(f"Ensemble created with {weights}")
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor(
            [m.stride.max() for m in model])).int()].stride  # max stride
        return model  # return ensemble

# torch 1.12.0 zhd-2024/09/03 14:22


def attempt_load_v2(weights, device='cpu', inplace=True, fuse=True):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(
            str(Path(str(w).strip().replace("'", ''))), map_location="cpu")
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(
            device).float()  # fp32 model

        # Model compatiablity updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict
        model.append(ckpt.fuse().eval() if fuse and hasattr(
            ckpt, "fuse") else ckpt.eval())  # model in eval mode
    # Module update
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)]*m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None
    # return model
    if len(model) == 1:
        return model[-1]
    # return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor(
        [m.stride.max() for m in model])).int()].stride  # max stride
    assert all(
        model[0].nc == m.nc for m in model), f"Models have different class counts:{[m.nc for m in model]}"
    return model


def load_engine(engine_path):
    if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(
            open(engine_path, "rb").read())

        return engine
