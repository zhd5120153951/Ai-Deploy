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


def attempt_load(weights, map_location=None, inplace=True):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    # torch.serialization.add_safe_globals([Model])
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(str(Path(str(w).strip().replace("'", ''))),
                          map_location=map_location, weights_only=False)  # load
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


def load_engine(engine_path):
    if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(
            open(engine_path, "rb").read())

        return engine
