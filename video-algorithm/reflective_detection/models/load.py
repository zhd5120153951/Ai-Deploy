from pathlib import Path

import torch
from torch import nn

from ai_common import setting


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = Path(str(w).strip().replace("'", ''))
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


def load_engine(engine1_path, engine2_path):
    if setting.TEST_FLAG == 0 and setting.REMOTE_DEBUG == 0:
        import tensorrt as trt
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine1 = runtime.deserialize_cuda_engine(open(engine1_path, "rb").read())
        engine2 = runtime.deserialize_cuda_engine(open(engine2_path, "rb").read())

        return engine1, engine2