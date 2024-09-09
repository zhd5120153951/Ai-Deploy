from pathlib import Path

import torch
from torch import nn

from models.common import Conv

from ai_common import setting


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = str(Path(str(w).strip().replace("'", '')))
        ckpt = torch.load(w, map_location=map_location)  # load
        model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def load_engine(engine1_path, engine2_path, engine3_path):
    if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine1_path = runtime.deserialize_cuda_engine(open(engine1_path, "rb").read())
        engine2_path = runtime.deserialize_cuda_engine(open(engine2_path, "rb").read())
        engine3_path = runtime.deserialize_cuda_engine(open(engine3_path, "rb").read())

        return engine1_path, engine2_path, engine3_path