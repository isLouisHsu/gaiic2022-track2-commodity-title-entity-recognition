import os
import torch
import json
import pickle
import logging
import glob
import numpy as np
import torch.nn as nn

logger = logging.getLogger(__name__)


def check_file(file_path):
    if not os.path.isfile(file_path):
        raise ValueError(f"File is not found here: {file_path}")
    return True


def is_file(file_path):
    if os.path.isfile(file_path):
        return True
    return False


def check_dir(dir_path):
    if not os.path.isdir(dir_path):
        raise ValueError(f"Directory is not found here: {dir_path}")
    return True


def find_all_files(dir_path):
    dir_path = os.path.expanduser(dir_path)
    files = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]
    logger.info(f"The number of files: {len(files)} , Direcory:{dir_path}")
    return


def create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        logger.info(f"Directory {dir_path} do not exist; creating...")


def save_pickle(data, file_path):
    with open(str(file_path), 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    with open(str(file_path), 'rb') as f:
        data = pickle.load(f)
    return data


def save_numpy(data, file_path):
    np.save(str(file_path), data)


def load_numpy(file_path):
    np.load(str(file_path))


def save_json(data, file_path):
    with open(str(file_path), 'w') as f:
        json.dump(data, f)


def load_json(file_path):
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.device):
            return str(obj)
        else:
            return super(_Encoder, self).default(obj)


def to_json_string(data):
    """Serializes this instance to a JSON string."""
    return json.dumps(data, indent=2, sort_keys=True, cls=_Encoder)


def json_to_text(file_path, data):
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')


def dict_to_text(file_path, data):
    with open(str(file_path), 'w') as fw:
        for key in sorted(data.keys()):
            fw.write("{} = {}\n".format(key, str(data[key])))


def save_model(model, file_path):
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    torch.save(state_dict, file_path)


def load_model(model, file_path, device=None):
    if check_file(file_path):
        print(f"loading model from {str(file_path)} .")
    state_dict = torch.load(file_path, map_location="cpu" if device is None else device)
    if isinstance(model, nn.DataParallel) or hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)


def save_jit_model(model, example_inputs, save_dir, dir_name=None):
    model.eval()
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_inputs=example_inputs, strict=False)
    if dir_name is None:
        save_dir = os.path.join(save_dir, 'save_model_jit_traced')
    else:
        save_dir = os.path.join(save_dir, dir_name)
    os.makedirs(save_dir, exist_ok=True)
    torch.jit.save(traced_model, os.path.join(save_dir, 'pytorch_model.ts'))
    return save_dir


def find_all_checkpoints(checkpoint_dir,
                         checkpoint_prefix='checkpoint',
                         checkpoint_name='pytorch_model.bin',
                         checkpoint_custom_names=None):
    '''
    获取模型保存路径下所有checkpoint模型路径，其中
    checkpoint_custom_names：表示自定义checkpoint列表
    '''
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(checkpoint_dir + "/**/" + checkpoint_name, recursive=True))
    )
    checkpoints = [x for x in checkpoints if checkpoint_prefix in x]
    if len(checkpoints) == 0:
        raise ValueError("No checkpoint found at : '{}'".format(checkpoint_dir))
    if checkpoint_custom_names is not None:
        if not isinstance(checkpoint_custom_names, list):
            checkpoint_custom_names = [checkpoint_custom_names]
        checkpoints = [x for x in checkpoints if x.split('/')[-1] in checkpoint_custom_names]
        logger.info(f"Successfully get checkpoints：{checkpoints}.")
    return checkpoints

import os
import time
import logging


class Logger:
    '''
    Base class for experiment loggers.
    日志模块
    '''

    def __init__(self, opts, log_file_level=logging.NOTSET):
        self.opts = opts
        self.log_file_level = log_file_level
        self.setup_logger()
        self.info = self.logger.info
        self.debug = self.logger.debug
        self.error = self.logger.error
        self.warning = self.logger.warning

    def setup_logger(self):
        log_file_path = self.setup_log_path()
        fmt = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        dmt = '%Y-%m-%d %H:%M:%S'
        log_format = logging.Formatter(fmt=fmt, datefmt=dmt)
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        self.logger.handlers = [console_handler]
        if log_file_path and log_file_path != '':
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(self.log_file_level)
            self.logger.addHandler(file_handler)

    def setup_time(self):
        local_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        return local_time

    def setup_log_path(self):
        log_time = self.setup_time()
        log_prefix = self.setup_prefix()
        log_file_name = f"{self.opts.task_name}-{self.opts.model_type}-" \
                        f"{self.opts.experiment_code}-{log_prefix}-{log_time}.log"
        log_file_path = os.path.join(self.opts.output_dir, log_file_name)
        return log_file_path

    def setup_prefix(self):
        if self.opts.do_train:
            return 'train'
        elif self.opts.do_eval:
            return 'eval'
        elif self.opts.do_predict:
            return 'predict'
        else:
            return ''



import sys
import time

class ProgressBar(object):
    '''
    自定义进度条
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step,info={'loss':20})
    '''

    def __init__(self,
                 n_total,
                 bar_width=50,
                 desc='Training',
                 num_epochs=None,
                 file=sys.stdout):
        self.desc = desc
        self.file = file
        self.width = bar_width
        self.n_total = n_total
        self.start_time = time.time()
        self.num_epochs = num_epochs

    def reset(self):
        """Method to reset internal variables."""
        self.start_time = time.time()

    def _time_info(self, now, current):
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = ('%d:%02d:%02d' %
                              (eta // 3600, (eta % 3600) // 60, eta % 60))
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta
            time_info = f' - ETA: {eta_format}'
        else:
            if time_per_unit >= 1:
                time_info = f' {time_per_unit:.1f}s/step'
            elif time_per_unit >= 1e-3:
                time_info = f' {time_per_unit * 1e3:.1f}ms/step'
            else:
                time_info = f' {time_per_unit * 1e6:.1f}us/step'
        return time_info

    def _bar(self, current):
        recv_per = current / self.n_total
        bar = f'[{self.desc}] {current}/{self.n_total} ['
        if recv_per >= 1: recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += '=' * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += '='
        bar += '.' * (self.width - prog_width)
        bar += ']'
        return bar

    def epoch(self, current_epoch):
        self.reset()
        self.file.write("\n")
        if (current_epoch is not None) and (self.num_epochs is not None):
            self.file.write(f"Epoch: {current_epoch}/{int(self.num_epochs)}")
            self.file.write("\n")

    def step(self, step, info={}):
        now = time.time()
        current = step + 1
        bar = self._bar(current)
        show_bar = f"\r{bar}" + self._time_info(now, current)
        if len(info) != 0:
            show_bar = f'{show_bar} ' + " [" + "-".join(
                [f' {key}={value:4f}  ' for key, value in info.items()]) + "]"
        if current >= self.n_total:
            show_bar += '\n'
            self.reset()
        self.file.write(show_bar)
        self.file.flush()

import os
import torch
import logging
import numpy as np


CHECKPOINT_DIR_PREFIX = 'checkpoint'
WEIGHTS_NAME = 'pytorch_model.bin'
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
VOCAB_NAME = 'vocab.json'


class ModelCheckpoint(object):
    '''
        Save the model after every epoch by monitoring a quantity.
    args:
        monitor: quantity to monitor. Default: ``eval_loss`` ,when save_best_only=True
        save_best_only: When `True`, always saves the best score model to a file `checpoint-best`. Default: ``False``.
    '''
    mode_dict = {'min': torch.lt, 'max': torch.gt}

    def __init__(self,
                 ckpt_dir,
                 mode='min',
                 monitor='eval_loss',
                 verbose=True,
                 save_best=False,
                 keys_to_ignore_on_save=[]
                 ):
        self.ckpt_dir = ckpt_dir
        self.monitor = monitor
        self.verbose = verbose
        self.save_best = save_best
        self.keys_to_ignore_on_save = keys_to_ignore_on_save

        if mode not in self.mode_dict:
            raise ValueError(f"mode: expected one of {', '.join(self.mode_dict.keys())}")
        self.monitor_op = self.mode_dict[mode]
        torch_inf = torch.tensor(np.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf
        self.init_save_dir()

    def init_save_dir(self):
        os.makedirs(self.ckpt_dir, exist_ok=True)
        prefix = f"{CHECKPOINT_DIR_PREFIX}-{self.monitor}"
        if self.save_best:
            self.save_ckpt_dir = os.path.join(self.ckpt_dir, f"{prefix}-best")
        else:
            self.save_ckpt_dir = os.path.join(self.ckpt_dir, prefix + '-{:.4f}-step-{}')

    def step(self, state, current=None):
        if current is not None and not isinstance(current, torch.Tensor): 
            current = torch.tensor(current)
        state['monitor'] = self.monitor
        state['score'] = current
        state['save_dir'] = self.save_ckpt_dir
        global_step = state['global_step']
        is_saving = False
        if current is None: # evaluate_during_training = False
            is_saving = True
        else:
            if not self.save_best:
                is_saving = True
                state['save_dir'] = self.save_ckpt_dir.format(state['score'], global_step)
            if self.monitor_op(current, self.best_score):  # best
                msg = (
                    f" Steps {global_step}: Metric {self.monitor} improved from {self.best_score:.4f} to {state['score']:.4f}"
                    f". New best score: {state['score']:.4f}"
                )
                logger.info(msg)
                self.best_score = current
                state['best_score'] = self.best_score
                is_saving = True
        if is_saving:
            for key in self.keys_to_ignore_on_save:
                if key in state:
                    state.pop(key)
            self.save_checkpoint(state)

    def save_checkpoint(self, state):
        os.makedirs(state['save_dir'], exist_ok=True)
        self._save_model(state)
        self._save_vocab(state)
        self._save_optimizer(state)
        self._save_scheduler(state)
        self._save_scaler(state)
        self._save_state(state)

    def _save_model(self, state):
        assert 'model' in state, "state['model'] does not exist."
        if self.verbose:
            logger.info("Saving model checkpoint to %s", state['save_dir'])
        model = state['model']
        if hasattr(model, 'save'):
            model.save(state['save_dir'])
        elif hasattr(model, 'save_pretrained'):
            model.save_pretrained(state['save_dir'])
        else:
            model_path = os.path.join(state['save_dir'], WEIGHTS_NAME)
            save_model(model, model_path)
        state.pop('model')

    def _save_vocab(self, state):
        if state.get('vocab', None):
            vocab = state['vocab']
            if hasattr(vocab, 'save_pretrained'):
                vocab.save_pretrained(state['save_dir'])
            else:
                file_path_name = os.path.join(state['save_dir'], VOCAB_NAME)
                if isinstance(vocab, dict):
                    json_to_text(file_path=file_path_name, data=vocab)
            state.pop('vocab')

    def _save_optimizer(self, state):
        if state.get('optimizer', None):
            file_path = os.path.join(state['save_dir'], OPTIMIZER_NAME)
            torch.save(state['optimizer'].state_dict(), file_path)
            state.pop('optimizer')

    def _save_scheduler(self, state):
        if state.get('scheduler', None):
            file_path = os.path.join(state['save_dir'], SCHEDULER_NAME)
            torch.save(state['scheduler'].state_dict(), file_path)
            state.pop('scheduler')

    def _save_scaler(self, state):
        if state.get('scaler', None):
            file_path = os.path.join(state['save_dir'], TRAINER_STATE_NAME)
            torch.save(state['scaler'].state_dict(), file_path)
            state.pop('scaler')

    def _save_state(self, state):
        file_path = os.path.join(state['save_dir'], TRAINER_STATE_NAME)
        torch.save(state, file_path)

import torch
import numpy as np
import logging

class EarlyStopping(object):
    '''
    Monitor a validation metric and stop training when it stops improving.

    Args:
        monitor: quantity to be monitored. Default: ``'eval_loss'``.
        min_delta: minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than `min_delta`, will count as no
            improvement. Default: ``0.0``.
        patience: number of validation epochs with no improvement
            after which training will be stopped. Default: ``10`.
        verbose: verbosity mode. Default: ``True``.
        mode: one of {min, max}. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing. Default: ``'min'``.
    '''

    mode_dict = {'min': torch.lt, 'max': torch.gt}

    def __init__(self,
                 min_delta=0,
                 patience=10,
                 verbose=True,
                 mode='min',
                 monitor='eval_loss',
                 save_state_path=None,
                 load_state_path=None
                 ):

        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.monitor = monitor
        self.wait_count = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.save_state_path = save_state_path

        if mode not in self.mode_dict:
            raise ValueError(f"mode: expected one of {', '.join(self.mode_dict.keys())}")
        self.monitor_op = self.mode_dict[mode]
        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

        if load_state_path is not None:
            self.load_state(load_state_path)
        if self.verbose:
            logger.info(f'EarlyStopping mode set to {mode} for monitoring {self.monitor}.')

    def save_state(self, save_path):
        state = {
            'wait_count': self.wait_count,
            'best_score': self.best_score,
            'patience': self.patience
        }
        torch.save(state, save_path)

    def load_state(self, state_path):
        state = torch.load(state_path)
        self.wait_count = state['wait_count']
        self.best_score = state['best_score']
        self.patience = state['patience']

    def step(self, current):
        if not isinstance(current, torch.Tensor): current = torch.tensor(current)
        if self.monitor_op(current, self.best_score):
            msg = (
                f" Metric {self.monitor} improved from {self.best_score:.4f} to {current:.4f}"
                f" New best score: {current:.3f}"
            )
            self.best_score = current
            self.wait_count = 0
            logger.info(msg)
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    msg = (f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                           f" Best score: {self.best_score:.3f}. Signaling Trainer to stop.")
                    logger.info(msg)
                if self.save_state_path is not None:
                    self.save_state(self.save_state_path)

import torch
import torch.nn as nn
from copy import deepcopy


class EMA(nn.Module):
    """ Model Exponential Moving Average V2

    Keep a moving average of everything in the model state_dict (parameters and buffers).
    V2 of this module is simpler, it does not match params/buffers based on name but simply
    iterates in order. It works with torchscript (JIT of full model).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage

    A smoothed version of the weights is necessary for some training schemes to perform well.
    E.g. Google's hyper-params for training MNASNet, MobileNet-V3, EfficientNet, etc that use
    RMSprop with a short 2.4-3 epoch decay period and slow LR decay rate of .96-.99 requires EMA
    smoothing of weights to match results. Pay attention to the decay constant you are using
    relative to your update count per epoch.

    To keep EMA from using GPU resources, set device='cpu'. This will save a bit of memory but
    disable validation of the EMA weights. Validation will have to be done manually in a separate
    process, or after the training stops converging.

    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class ProcessBase(object):
    """ 用于处理单个example """

    def __call__(self, example):
        raise NotImplementedError('Method [__call__] should be implemented.')

def get_spans_bios(tags, id2label=None):
    """Gets entities from sequence.
    note: BIOS
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'S-LOC']
        >>> get_spans_bios(tags)
        # output: [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunks.append(chunk)
            chunk = (-1, -1, -1)
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


def get_spans_bio(tags, id2label=None):
    """Gets entities from sequence.
    Args:
        tags (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        >>> tags = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_spans_bio(tags)
        # output [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == len(tags) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


TYPE_TO_SCHEME = {
    "BIO": get_spans_bio,
    "BIOS": get_spans_bios,
}


def get_scheme(scheme_type):
    if scheme_type not in TYPE_TO_SCHEME:
        msg = ("There were expected keys in the `TYPE_TO_SCHEME`: "
               f"{', '.join(list(TYPE_TO_SCHEME.keys()))}, "
               f"but get {scheme_type}."
               )
        raise TypeError(msg)
    scheme_function = TYPE_TO_SCHEME[scheme_type]
    return scheme_function

import numpy as np
import warnings
from typing import *

def _warn_prf(average, modifier, msg_start, result_size):
    axis0, axis1 = 'sample', 'label'
    if average == 'samples':
        axis0, axis1 = axis1, axis0
    msg = ('{0} ill-defined and being set to 0.0 {{0}} '
           'no {1} {2}s. Use `zero_division` parameter to control'
           ' this behavior.'.format(msg_start, modifier, axis0))
    if result_size == 1:
        msg = msg.format('due to')
    else:
        msg = msg.format('in {0}s with'.format(axis1))
    warnings.warn(msg, UserWarning, stacklevel=2)


def _prf_divide(numerator, denominator, metric,
                modifier, average, warn_for, zero_division='warn'):
    """Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or 1 (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining
    an appropriate warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator
    if not np.any(mask):
        return result
    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ['warn', 0] else 1.0
    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != 'warn' or metric not in warn_for:
        return result
    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."
    if metric in warn_for and 'f-score' in warn_for:
        msg_start = '{0} and F-score are'.format(metric.title())
    elif metric in warn_for:
        msg_start = '{0} is'.format(metric.title())
    elif 'f-score' in warn_for:
        msg_start = 'F-score is'
    else:
        return result
    _warn_prf(average, modifier, msg_start, len(result))
    return result


def check_consistent_length(y_true: List[List[str]], y_pred: List[List[str]]):
    """Check that all arrays have consistent first and second dimensions.

    Checks whether all objects in arrays have the same shape or length.

    Args:
        y_true : 2d array.
        y_pred : 2d array.
    """
    len_true = list(map(len, y_true))
    len_pred = list(map(len, y_pred))
    is_list = set(map(type, y_true)) | set(map(type, y_pred))
    if not is_list == {list}:
        raise TypeError('Found input variables without list of list.')

    if len(y_true) != len(y_pred) or len_true != len_pred:
        message = 'Found input variables with inconsistent numbers of samples:\n{}\n{}'.format(len_true, len_pred)
        raise ValueError(message)

import numpy as np
from typing import *
from collections import defaultdict

PER_CLASS_SCORES = Tuple[List[float], List[float], List[float], List[int]]
AVERAGE_SCORES = Tuple[float, float, float, int]
SCORES = Union[PER_CLASS_SCORES, AVERAGE_SCORES]


def _precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                     y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                     *,
                                     average: Optional[str] = None,
                                     warn_for=('precision', 'recall', 'f-score'),
                                     beta: float = 1.0,
                                     sample_weight: Optional[List[int]] = None,
                                     zero_division: str = 'warn',
                                     scheme: Optional[Type[Any]] = None,
                                     suffix: bool = False,
                                     extract_tp_actual_correct: Callable = None) -> SCORES:
    if beta < 0:
        raise ValueError('beta should be >=0 in the F-beta score')

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options:
        raise ValueError('average has to be one of {}'.format(average_options))

    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred, suffix, scheme)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #
    beta2 = beta ** 2

    # Divide, and on zero-division, set scores and/or warn according to
    # zero_division:
    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )
    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=warn_for,
        zero_division=zero_division
    )

    # warn for f-score only if zero_division is warn, it is in warn_for
    # and BOTH prec and rec are ill-defined
    if zero_division == 'warn' and ('f-score',) == warn_for:
        if (pred_sum[true_sum == 0] == 0).any():
            _warn_prf(
                average, 'true nor predicted', 'F-score is', len(true_sum)
            )

    # if tp == 0 F will be 1 only if all predictions are zero, all labels are
    # zero, and zero_division=1. In all other case, 0
    if np.isposinf(beta):
        f_score = recall
    else:
        denom = beta2 * precision + recall

        denom[denom == 0.] = 1  # avoid division by 0
        f_score = (1 + beta2) * precision * recall / denom

    # Average the results
    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            zero_division_value = 0.0 if zero_division in ['warn', 0] else 1.0
            # precision is zero_division if there are no positive predictions
            # recall is zero_division if there are no positive labels
            # fscore is zero_division if all labels AND predictions are
            # negative
            return (zero_division_value if pred_sum.sum() == 0 else 0.0,
                    zero_division_value,
                    zero_division_value if pred_sum.sum() == 0 else 0.0,
                    sum(true_sum))

    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None
    if average is not None:
        precision = np.average(precision, weights=weights)
        recall = np.average(recall, weights=weights)
        f_score = np.average(f_score, weights=weights)
        true_sum = sum(true_sum)

    return precision, recall, f_score, true_sum


def precision_recall_fscore_support(y_true: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    y_pred: Union[List[List[str]], List[List[Tuple[int, int, Any]]]],
                                    *,
                                    average: Optional[str] = None,
                                    labels: Optional[List[str]] = None,
                                    warn_for=('precision', 'recall', 'f-score'),
                                    beta: float = 1.0,
                                    sample_weight: Optional[List[int]] = None,
                                    zero_division: str = 'warn',
                                    suffix: bool = False,
                                    schema: str = "BIO",
                                    ) -> SCORES:
    """Compute precision, recall, F-measure and support for each class.
    Args:
        target : 2d array. Ground truth (correct) target values.
        preds : 2d array. Estimated targets as returned by a tagger.
        beta : float, 1.0 by default
            The strength of recall versus precision in the F-score.
        average : string, [None (default), 'micro', 'macro', 'weighted']
            If ``None``, the scores for each class are returned. Otherwise, this
            determines the type of averaging performed on the data:
            ``'micro'``:
                Calculate metrics globally by counting the total true positives,
                false negatives and false positives.
            ``'macro'``:
                Calculate metrics for each label, and find their unweighted
                mean.  This does not take label imbalance into account.
            ``'weighted'``:
                Calculate metrics for each label, and find their average weighted
                by support (the number of true instances for each label). This
                alters 'macro' to account for label imbalance; it can result in an
                F-score that is not between precision and recall.
        warn_for : tuple or set, for internal use
            This determines which warnings will be made in the case that this
            function is being used to return only one of its metrics.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.
        zero_division : "warn", 0 or 1, default="warn"
            Sets the value to return when there is a zero division:
               - recall: when there are no positive labels
               - precision: when there are no positive predictions
               - f-score: both
            If set to "warn", this acts as 0, but warnings are also raised.
        suffix : bool, False by default.
    Returns:
        precision : float (if average is not None) or array of float, shape = [n_unique_labels]
        recall : float (if average is not None) or array of float, , shape = [n_unique_labels]
        fbeta_score : float (if average is not None) or array of float, shape = [n_unique_labels]
        support : int (if average is not None) or array of int, shape = [n_unique_labels]
            The number of occurrences of each label in ``y_true``.
    Examples:
        >>> from torchblocks.metrics.sequence_labeling.precision_recall_fscore import precision_recall_fscore_support
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_recall_fscore_support(y_true, y_pred, average='macro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='micro')
        (0.5, 0.5, 0.5, 2)
        >>> precision_recall_fscore_support(y_true, y_pred, average='weighted')
        (0.5, 0.5, 0.5, 2)
        It is possible to compute per-label precisions, recalls, F1-scores and
        supports instead of averaging:
        >>> precision_recall_fscore_support(y_true, y_pred, average=None)
        (array([0., 1.]), array([0., 1.]), array([0., 1.]), array([1, 1]))
    Notes:
        When ``true positive + false positive == 0``, precision is undefined;
        When ``true positive + false negative == 0``, recall is undefined.
        In such cases, by default the metric will be set to 0, as will f-score,
        and ``UndefinedMetricWarning`` will be raised. This behavior can be
        modified with ``zero_division``.
    """

    def extract_tp_actual_correct(y_true, y_pred, suffix, *args):
        entities_true = defaultdict(set)
        entities_pred = defaultdict(set)
        if len(y_pred[0]) > 0 and isinstance(y_pred[0][0], str):
            check_consistent_length(y_true, y_pred)
            get_entities = get_scheme(scheme_type=schema)
            for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
                for type_name, start, end in get_entities(y_t):
                    entities_true[type_name].add((i, start, end))
                for type_name, start, end in get_entities(y_p):
                    entities_pred[type_name].add((i, start, end))
        else:
            for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
                for start, end, type_name in y_t:
                    entities_true[type_name].add((i, start, end))
                for start, end, type_name in y_p:
                    entities_pred[type_name].add((i, start, end))
        if labels is not None:
            entities_true = {k: v for k, v in entities_true.items() if k in labels}
            entities_pred = {k: v for k, v in entities_pred.items() if k in labels}
        target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))
        tp_sum = np.array([], dtype=np.int32)
        pred_sum = np.array([], dtype=np.int32)
        true_sum = np.array([], dtype=np.int32)
        for type_name in target_names:
            entities_true_type = entities_true.get(type_name, set())
            entities_pred_type = entities_pred.get(type_name, set())
            tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
            pred_sum = np.append(pred_sum, len(entities_pred_type))
            true_sum = np.append(true_sum, len(entities_true_type))
        return pred_sum, tp_sum, true_sum

    precision, recall, f_score, true_sum = _precision_recall_fscore_support(
        y_true, y_pred,
        average=average,
        warn_for=warn_for,
        beta=beta,
        sample_weight=sample_weight,
        zero_division=zero_division,
        scheme=None,
        suffix=suffix,
        extract_tp_actual_correct=extract_tp_actual_correct
    )
    return precision, recall, f_score, true_sum

class Metric:
    """Store the average and current value for a set of metrics.
    """
    def update(self, preds, target):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    def reset(self):
        pass


import pandas as pd


class SequenceLabelingScore(Metric):

    def __init__(self, labels, schema=None, average="micro"):
        self.labels = labels
        self.schema = schema
        self.average = average
        self.reset()

    def update(self, preds, target):
        self.preds.extend(preds)
        self.target.extend(target)

    def value(self):
        columns = ["label", "precision", "recall", "f1", "support"]
        values = []
        for label in [self.average] + sorted(self.labels):
            p, r, f, s = precision_recall_fscore_support(
                self.target, self.preds, average=self.average, schema=self.schema,
                labels=None if label == self.average else [label])
            values.append([label, p, r, f, s])
        df = pd.DataFrame(values, columns=columns)
        f1 = df[df['label'] == self.average]['f1'].item()
        return {
            "df": df, f"f1_{self.average}": f1,  # for monitor
        }

    def name(self):
        return "seqTag"

    def reset(self):
        self.preds = []
        self.target = []

import torch
from torch import nn

class ConditionalLayerNorm(nn.Module):
    def __init__(self,
                 normalized_shape,
                 cond_shape,
                 eps=1e-12):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(normalized_shape))
        self.bias = nn.Parameter(torch.Tensor(normalized_shape))
        self.weight_dense = nn.Linear(cond_shape, normalized_shape, bias=False)
        self.bias_dense = nn.Linear(cond_shape, normalized_shape, bias=False)

        self.reset_weight_and_bias()

    def reset_weight_and_bias(self):
        """
        此处初始化的作用是在训练开始阶段不让 conditional layer norm 起作用
        """
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.zeros_(self.weight_dense.weight)
        nn.init.zeros_(self.bias_dense.weight)

    def forward(self, inputs, cond=None):
        assert cond is not None, 'Conditional tensor need to input when use conditional layer norm'
        cond = torch.unsqueeze(cond, 1)  # (b, 1, h*2)
        weight = self.weight_dense(cond) + self.weight  # (b, 1, h)
        bias = self.bias_dense(cond) + self.bias  # (b, 1, h)
        mean = torch.mean(inputs, dim=-1, keepdim=True)  # （b, s, 1）
        outputs = inputs - mean  # (b, s, h)
        variance = torch.mean(outputs ** 2, dim=-1, keepdim=True)
        std = torch.sqrt(variance + self.eps)  # (b, s, 1)
        outputs = outputs / std  # (b, s, h)
        outputs = outputs * weight + bias
        return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Softmax and sigmoid focal loss
    """

    def __init__(self,
                 num_labels,
                 gamma=2.0,
                 alpha=0.25,
                 epsilon=1.e-9,
                 reduction='mean',
                 activation_type='softmax'
                 ):

        super(FocalLoss, self).__init__()
        self.num_labels = num_labels
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.activation_type = activation_type
        self.reduction = reduction

    def forward(self, preds, target):
        """
        Args:
            logits: model's output, shape of [batch_size, num_cls]
            target: ground truth labels, shape of [batch_size]
        Returns:
            shape of [batch_size]
        """
        if self.activation_type == 'softmax':
            idx = target.view(-1, 1).long()
            one_hot_key = torch.zeros(idx.size(0), self.num_labels, dtype=torch.float32, device=idx.device)
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            logits = F.softmax(preds, dim=-1)
            loss = -self.alpha * one_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss = loss.sum(1)
        elif self.activation_type == 'sigmoid':
            multi_hot_key = target
            logits = F.sigmoid(preds)
            zero_hot_key = 1 - multi_hot_key
            loss = -self.alpha * multi_hot_key * torch.pow((1 - logits), self.gamma) * (logits + self.epsilon).log()
            loss += -(1 - self.alpha) * zero_hot_key * torch.pow(logits, self.gamma) * (1 - logits + self.epsilon).log()
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        return loss


class FocalCosineLoss(nn.Module):
    """Implementation Focal cosine loss.

    [Data-Efficient Deep Learning Method for Image Classification
    Using Data Augmentation, Focal Cosine Loss, and Ensemble](https://arxiv.org/abs/2007.07805).

    Source : <https://www.kaggle.com/c/cassava-leaf-disease-classification/discussion/203271>
    """

    def __init__(self, alpha=1, gamma=2, xent=0.1, reduction="mean"):
        """Constructor for FocalCosineLoss.
        """
        super(FocalCosineLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.xent = xent
        self.reduction = reduction

    def forward(self, logits, target):
        """Forward Method."""
        cosine_loss = F.cosine_embedding_loss(
            logits,
            torch.nn.functional.one_hot(target, num_classes=logits.size(-1)),
            torch.tensor([1], device=target.device),
            reduction=self.reduction,
        )

        cent_loss = F.cross_entropy(F.normalize(logits), target, reduction="none")
        pt = torch.exp(-cent_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * cent_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)
        return cosine_loss + self.xent * focal_loss

import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCE, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, target):
        c = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        loss_1 = loss * self.eps / c
        loss_2 = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)
        return loss_1 + (1 - self.eps) * loss_2

import torch
from torch import nn
from torch.nn import functional as F

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=self.num_classes).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1

class Poly1FocalLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 epsilon: float = 1.0,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 reduction: str = "none",
                 weight: torch.Tensor = None,
                 pos_weight: torch.Tensor = None,
                 label_is_onehot: bool = False):
        """
        Create instance of Poly1FocalLoss
        :param num_classes: number of classes
        :param epsilon: poly loss epsilon
        :param alpha: focal loss alpha
        :param gamma: focal loss gamma
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to binary Cross-Entropy loss
        :param label_is_onehot: set to True if labels are one-hot encoded
        """
        super(Poly1FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight
        self.label_is_onehot = label_is_onehot

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: output of neural netwrok of shape [N, num_classes] or [N, num_classes, ...]
        :param labels: ground truth tensor of shape [N] or [N, ...] with class ids if label_is_onehot was set to False, otherwise
            one-hot encoded tensor of same shape as logits
        :return: poly focal loss
        """
        # focal loss implementation taken from
        # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py

        p = torch.sigmoid(logits)

        if not self.label_is_onehot:
            # if labels are of shape [N]
            # convert to one-hot tensor of shape [N, num_classes]
            if labels.ndim == 1:
                labels = F.one_hot(labels, num_classes=self.num_classes)

            # if labels are of shape [N, ...] e.g. segmentation task
            # convert to one-hot tensor of shape [N, num_classes, ...]
            else:
                labels = F.one_hot(labels.unsqueeze(1), self.num_classes).transpose(1, -1).squeeze_(-1)

        labels = labels.to(device=logits.device,
                           dtype=logits.dtype)

        ce_loss = F.binary_cross_entropy_with_logits(input=logits,
                                                     target=labels,
                                                     reduction="none",
                                                     weight=self.weight,
                                                     pos_weight=self.pos_weight)
        pt = labels * p + (1 - labels) * (1 - p)
        FL = ce_loss * ((1 - pt) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
            FL = alpha_t * FL

        poly1 = FL + self.epsilon * torch.pow(1 - pt, self.gamma + 1)

        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()

        return poly1

# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for working with package versions
"""

import operator
import re
import sys
from typing import Optional

from packaging import version


# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


ops = {
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _compare_versions(op, got_ver, want_ver, requirement, pkg, hint):
    if got_ver is None:
        raise ValueError("got_ver is None")
    if want_ver is None:
        raise ValueError("want_ver is None")
    if not ops[op](version.parse(got_ver), version.parse(want_ver)):
        raise ImportError(
            f"{requirement} is required for a normal functioning of this module, but found {pkg}=={got_ver}.{hint}"
        )


def require_version(requirement: str, hint: Optional[str] = None) -> None:
    """
    Perform a runtime check of the dependency versions, using the exact same syntax used by pip.

    The installed module version comes from the `site-packages` dir via `importlib_metadata`.

    Args:
        requirement (:obj:`str`): pip style definition, e.g.,  "tokenizers==0.9.4", "tqdm>=4.27", "numpy"
        hint (:obj:`str`, `optional`): what suggestion to print in case of requirements not being met

    Example::

       require_version("pandas>1.1.2")
       require_version("numpy>1.18.5", "this is important to have for whatever reason")

    """

    hint = f"\n{hint}" if hint is not None else ""

    # non-versioned check
    if re.match(r"^[\w_\-\d]+$", requirement):
        pkg, op, want_ver = requirement, None, None
    else:
        match = re.findall(r"^([^!=<>\s]+)([\s!=<>]{1,2}.+)", requirement)
        if not match:
            raise ValueError(
                f"requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}"
            )
        pkg, want_full = match[0]
        want_range = want_full.split(",")  # there could be multiple requirements
        wanted = {}
        for w in want_range:
            match = re.findall(r"^([\s!=<>]{1,2})(.+)", w)
            if not match:
                raise ValueError(
                    f"requirement needs to be in the pip package format, .e.g., package_a==1.23, or package_b>=1.23, but got {requirement}"
                )
            op, want_ver = match[0]
            wanted[op] = want_ver
            if op not in ops:
                raise ValueError(f"{requirement}: need one of {list(ops.keys())}, but got {op}")

    # special case
    if pkg == "python":
        got_ver = ".".join([str(x) for x in sys.version_info[:3]])
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
        return

    # check if any version is installed
    try:
        got_ver = importlib_metadata.version(pkg)
    except importlib_metadata.PackageNotFoundError:
        raise importlib_metadata.PackageNotFoundError(
            f"The '{requirement}' distribution was not found and is required by this application. {hint}"
        )

    # check that the right version is installed if version number or a range was provided
    if want_ver is not None:
        for op, want_ver in wanted.items():
            _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)


def require_version_core(requirement):
    """require_version wrapper which emits a core-specific hint on failure"""
    hint = "Try: pip install transformers -U or pip install -e '.[dev]' if you're working with git master"
    return require_version(requirement, hint)


import torch
import math
from torch import nn
from torch.optim.optimizer import Optimizer
from typing import Callable, Iterable, Optional, Tuple, Union

class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in `Decoupled Weight Decay Regularization
    <https://arxiv.org/abs/1711.05101>`__.

    Parameters:
        params (:obj:`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (:obj:`float`, `optional`, defaults to 1e-3):
            The learning rate to use.
        betas (:obj:`Tuple[float,float]`, `optional`, defaults to (0.9, 0.999)):
            Adam's betas parameters (b1, b2).
        eps (:obj:`float`, `optional`, defaults to 1e-6):
            Adam's epsilon for numerical stability.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            Decoupled weight decay to apply.
        correct_bias (:obj:`bool`, `optional`, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use :obj:`False`).
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
    ):
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0[")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0[")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (:obj:`Callable`, `optional`): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))

        return loss

import torch


class FGM(object):
    def __init__(self, model, emb_name, epsilon=1.0):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


class PGD(object):
    def __init__(self, model, emb_name, epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}


import torch

class AWP(object):
    """ [Adversarial weight perturbation helps robust generalization](https://arxiv.org/abs/2004.05884)
    """
    def __init__(
        self,
        model,
        emb_name="weight",
        epsilon=0.001,
        alpha=1.0,
    ):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.param_backup = {}
        self.param_backup_eps = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        if self.alpha == 0: return
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.emb_name in name:
                # save
                if is_first_attack:
                    self.param_backup[name] = param.data.clone()
                    grad_eps = self.epsilon * param.abs().detach()
                    self.param_backup_eps[name] = (
                        self.param_backup[name] - grad_eps,
                        self.param_backup[name] + grad_eps,
                    )
                # attack
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.alpha * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(
                            param.data, 
                            self.param_backup_eps[name][0]
                        ), 
                        self.param_backup_eps[name][1]
                    )

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.param_backup:
                param.data = self.param_backup[name]
        self.param_backup = {}
        self.param_backup_eps = {}

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
        self.grad_backup = {}

import importlib.util
def is_apex_available():
    return importlib.util.find_spec("apex") is not None

import math
import torch
import logging
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR



def get_constant_schedule(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_polynomial_decay_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.

    """

    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining ** power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER = {
    'constant': get_constant_schedule,
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant_with_warmup': get_constant_schedule_with_warmup,
    'cosine_with_restarts': get_cosine_with_hard_restarts_schedule_with_warmup
}


def get_lr_scheduler(scheduler_type):
    scheduler_function = TYPE_TO_SCHEDULER[scheduler_type]
    return scheduler_function


def check_object_type(object, check_type, name):
    if not isinstance(object, check_type):
        raise TypeError(f"The type of {name} must be {check_type}, but got {type(object)}.")


import os
import random
import numpy as np
import torch
import logging


def select_seed_randomly(min_seed_value=0, max_seed_value=1024):
    seed = random.randint(min_seed_value, max_seed_value)
    logger.warning((f"No seed found, seed set to {seed}"))
    return int(seed)


def seed_everything(seed=None,verbose=True):
    '''
    init random seed for random functions in numpy, torch, cuda and cudnn
    Args:
        seed (int): random seed
        reproducibility (bool): Whether to require reproducibility
    '''
    if seed is None: seed = select_seed_randomly()
    if verbose:
        logger.info(f"Global seed set to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if verbose:
            logger.info("cudnn is enabled.")


class AverageMeter(object):
    '''
    computes and stores the average and current value
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        """Method to reset all the internal variables."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from collections import defaultdict

plt.switch_backend('agg')  # 防止ssh上绘图问题

FILE_NAME = 'training_info.json'


class FileWriter:

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.scale_dicts = defaultdict(list)
        create_dir(self.log_dir)

    def add_scalar(self, tag, scalar_value, global_step=None):
        if global_step is not None:
            global_step = int(global_step)
        _dict = {tag: scalar_value, 'step': global_step}
        self.scale_dicts[tag].append(_dict)

    def save(self, plot=True):
        save_path = os.path.join(self.log_dir, FILE_NAME)
        save_json(data=self.scale_dicts, file_path=save_path)
        if plot:
            self.plot()

    def close(self):
        pass

    def plot(self):
        keys = list(self.scale_dicts.keys())
        for key in keys:
            values = self.scale_dicts[key]
            name = key.split("/")[-1] if "/" in key else key
            png_file = os.path.join(self.log_dir, f"{name}")

            values = sorted(values, key=lambda x: x['step'])
            x = [i['step'] for i in values]
            y = [i[key] for i in values]

            plt.style.use("ggplot")
            fig = plt.figure(figsize=(15, 5), facecolor='w')
            ax = fig.add_subplot(111)
            if "eval_" in name:
                y = [round(float(x), 2) for x in y]
            ax.plot(x, y, label=name)
            if key == 'train_lr':
                # 科学计数法显示
                ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            ax.legend()
            plt.xlabel("Step #")
            plt.ylabel(name)
            plt.title(f"Training {name} [Step {x[-1]}]")
            plt.savefig(png_file)
            plt.close()


import os
import sys
import math
import torch
import warnings
import pandas as pd
import torch.nn as nn
from argparse import Namespace
from packaging import version

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

if not sys.warnoptions:
    warnings.simplefilter("ignore")

_is_native_amp_available = False
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

try:
    from torch.utils.tensorboard import SummaryWriter

    _has_tensorboard = True
except ImportError:
    try:
        from tensorboardX import SummaryWriter

        _has_tensorboard = True
    except ImportError:
        _has_tensorboard = False


class TrainerBase:
    """Base class for iterative trainer."""
    keys_to_ignore_on_gpu = []  # batch不存放在gpu中的变量，比如'input_length’
    keys_to_ignore_on_result_save = ['input_ids', 'token_type_ids']  # eval和predict结果不存储的变量
    keys_to_ignore_on_checkpoint_save = []  # checkpoint中不存储的模块，比如'optimizer'

    def __init__(self,
                 opts,
                 model,
                 metrics,
                 logger,
                 optimizer=None,
                 scheduler=None,
                 adv_model=None,
                 model_checkpoint=None,
                 early_stopping=None,
                 **kwargs):
        self.opts = opts
        self.model = model
        self.metrics = metrics
        self.logger = logger
        self.scheduler = scheduler
        self.global_step = 0
        self.device_num = getattr(opts, 'device_num', 0)
        self.warmup_steps = getattr(opts, 'warmup_steps', 0)
        self.num_train_epochs = getattr(opts, "num_train_epochs", 3)
        self.device = getattr(opts, 'device', torch.device("cpu"))
        self.max_grad_norm = getattr(opts, 'max_grad_norm', 0.0)
        self.warmup_proportion = getattr(opts, 'warmup_proportion', 0.1)
        self.gradient_accumulation_steps = getattr(opts, "gradient_accumulation_steps", 1)
        self.prefix = "_".join([opts.model_type, opts.task_name, opts.experiment_code])
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.build_writer()
        self.build_mixed_precision()
        if not isinstance(self.metrics, list):
            self.metrics = [self.metrics]
        check_object_type(object=self.metrics, check_type=list, name='metric')
        check_object_type(object=self.model, check_type=nn.Module, name='model')
        check_object_type(object=self.opts, check_type=Namespace, name='self.opts')
        check_object_type(object=self.logger, check_type=Logger, name='self.logger')
        # EMA
        if opts.ema_enable:
            self.logger.info('Using EMA')
            self.model_ema = EMA(model=self.model,
                                 decay=opts.ema_decay,
                                 device='cpu' if opts.model_ema_force_cpu else None)
        # Adversarial training
        if opts.adv_enable:
            msg = f"Using Adversarial training and type: {opts.adv_type}"
            self.logger.info(msg)
            self.adv_model = adv_model
            if adv_model is None:
                self.adv_model = self.build_adv_model()
        # optimizer
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = self.build_optimizer(model)
        # checkpoint
        self.model_checkpoint = model_checkpoint
        if model_checkpoint is None:
            self.model_checkpoint = ModelCheckpoint(
                mode=opts.checkpoint_mode,
                monitor=opts.checkpoint_monitor,
                ckpt_dir=opts.output_dir,
                verbose=opts.checkpoint_verbose,
                save_best=opts.checkpoint_save_best,
                keys_to_ignore_on_save=self.keys_to_ignore_on_checkpoint_save
            )
        # earlystopping
        self.early_stopping = early_stopping
        if early_stopping is None and opts.earlystopping_patience > 0:
            self.early_stopping = EarlyStopping(
                mode=opts.earlystopping_mode,
                patience=opts.earlystopping_patience,
                monitor=opts.earlystopping_monitor,
                save_state_path=opts.earlystopping_save_state_path,
                load_state_path=opts.earlystopping_load_state_path
            )

    def build_mixed_precision(self):
        # Mixed precision setup
        self.use_apex = False
        self.use_amp = False
        self.fp16_backend = None
        if self.opts.fp16:
            if self.opts.fp16_backend == "auto":
                self.fp16_backend = "amp" if _is_native_amp_available else "apex"
            else:
                self.fp16_backend = self.opts.fp16_backend
            self.logger.info(f"Using {self.fp16_backend} fp16 backend")
            if self.fp16_backend == "amp":
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                if not is_apex_available():
                    msg = ("Using FP16 with APEX but APEX is not installed, "
                           "please refer to https://www.github.com/nvidia/apex.")
                    raise ImportError(msg)
                self.use_apex = True

    def build_adv_model(self):
        adv_model = None
        if self.opts.adv_type == 'fgm':
            adv_model = FGM(self.model,
                            emb_name=self.opts.adv_name,
                            epsilon=self.opts.adv_epsilon)
        elif self.opts.adv_type == 'pgd':
            adv_model = PGD(self.model,
                            emb_name=self.opts.adv_name,
                            epsilon=self.opts.adv_epsilon,
                            alpha=self.opts.adv_alpha)
        elif self.opts.adv_type == "awp":
            adv_model = AWP(self.model,
                            emb_name=self.opts.adv_name,
                            epsilon=self.opts.adv_epsilon,
                            alpha=self.opts.adv_alpha)
        return adv_model

    def build_record_tracker(self, **kwargs):
        '''
        build record object
        '''
        self.records = {}
        self.records['result'] = {}
        self.records['loss_meter'] = AverageMeter()
        for key, value in kwargs.items():
            if key not in self.records:
                self.records[key] = value

    def reset_metrics(self):
        for metric in self.metrics:
            if hasattr(metric, 'reset'):
                metric.reset()

    def _param_optimizer(self, params, learning_rate, no_decay, weight_decay):
        _params = [
            {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay,
             'lr': learning_rate},
            {'params': [p for n, p in params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0,
             'lr': learning_rate},
        ]
        return _params

    def build_model_param_optimizer(self, model):
        '''
        若需要对不同模型赋予不同学习率，则指定`base_model_name`,
        在`transformer`模块中，默认为`base_model_name=`base_model`.
        对于base_model使用learning_rate，
        其余统一使用other_learning_rate
        '''
        no_decay = ["bias", 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        if hasattr(model, self.opts.base_model_name) and self.opts.other_learning_rate != 0.0:
            msg = (f"The initial learning rate for model params : {self.opts.learning_rate} ,"
                   f"and {self.opts.other_learning_rate}"
                   )
            self.logger.info(msg)
            base_model = getattr(model, self.opts.base_model_name)
            base_model_param = list(base_model.named_parameters())
            base_model_param_ids = [id(p) for n, p in base_model_param]
            other_model_param = [(n, p) for n, p in model.named_parameters() if
                                 id(p) not in base_model_param_ids]
            optimizer_grouped_parameters.extend(
                self._param_optimizer(base_model_param, self.opts.learning_rate, no_decay, self.opts.weight_decay))
            optimizer_grouped_parameters.extend(
                self._param_optimizer(other_model_param, self.opts.other_learning_rate, no_decay,
                                      self.opts.weight_decay))
        else:
            all_model_param = list(model.named_parameters())
            optimizer_grouped_parameters.extend(
                self._param_optimizer(all_model_param, self.opts.learning_rate, no_decay, self.opts.weight_decay))
        return optimizer_grouped_parameters

    def build_optimizer(self, model):
        '''
        Setup the optimizer.
        '''
        optimizer_grouped_parameters = self.build_model_param_optimizer(model)
        optimizer = AdamW(params=optimizer_grouped_parameters,
                          lr=self.opts.learning_rate,
                          eps=self.opts.adam_epsilon,
                          betas=(self.opts.adam_beta1, self.opts.adam_beta2),
                          weight_decay=self.opts.weight_decay)
        return optimizer

    def build_warmup_steps(self, num_training_steps):
        """
        Get number of steps used for a linear warmup.
        """
        if self.warmup_proportion < 0 or self.warmup_proportion > 1:
            raise ValueError("warmup_proportion must lie in range [0,1]")
        elif self.warmup_proportion > 0 and self.warmup_steps > 0:
            msg = ("Both warmup_ratio and warmup_steps given, "
                   "warmup_steps will override any effect of warmup_ratio during training")
            self.logger.info(msg)
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(
                num_training_steps * self.warmup_proportion)
        )
        return warmup_steps

    def build_lr_scheduler(self, num_training_steps):
        '''
        the learning rate scheduler.
        '''
        scheduler_function = get_lr_scheduler(self.opts.scheduler_type)
        warmup_steps = self.build_warmup_steps(num_training_steps)
        scheduler = scheduler_function(optimizer=self.optimizer,
                                       num_warmup_steps=warmup_steps,
                                       num_training_steps=num_training_steps)
        return scheduler

    def build_train_dataloader(self, train_data):
        '''
        Load train datasets
        '''
        if isinstance(train_data, DataLoader):
            return train_data
        elif isinstance(train_data, Dataset):
            batch_size = self.opts.per_gpu_train_batch_size * max(1, self.device_num)
            sampler = RandomSampler(train_data) if not hasattr(train_data, 'sampler') else train_data.sampler
            collate_fn = train_data.collate_fn if hasattr(train_data, 'collate_fn') else None
            data_loader = DataLoader(train_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     drop_last=self.opts.drop_last,
                                     num_workers=self.opts.num_workers)
            return data_loader
        else:
            raise TypeError("train_data type{} not support".format(type(train_data)))

    def build_eval_dataloader(self, dev_data):
        '''
        Load eval datasets
        '''
        if isinstance(dev_data, DataLoader):
            return dev_data
        elif isinstance(dev_data, Dataset):
            batch_size = self.opts.per_gpu_eval_batch_size * max(1, self.device_num)
            sampler = SequentialSampler(dev_data) if not hasattr(dev_data, 'sampler') else dev_data.sampler
            collate_fn = dev_data.collate_fn if hasattr(dev_data, 'collate_fn') else None
            data_loader = DataLoader(dev_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     num_workers=self.opts.num_workers)
            return data_loader
        else:
            raise TypeError("dev_data type{} not support".format(type(dev_data)))

    def build_test_dataloader(self, test_data):
        '''
        Load test datasets
        '''
        if isinstance(test_data, DataLoader):
            return test_data
        elif isinstance(test_data, Dataset):
            batch_size = self.opts.per_gpu_test_batch_size * max(1, self.device_num)
            sampler = SequentialSampler(test_data) if not hasattr(test_data, 'sampler') else test_data.sampler
            collate_fn = test_data.collate_fn if hasattr(test_data, 'collate_fn') else None
            data_loader = DataLoader(test_data,
                                     sampler=sampler,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     num_workers=self.opts.num_workers)
            return data_loader
        else:
            raise TypeError("test_data type{} not support".format(type(test_data)))

    def build_batch_inputs(self, batch):
        '''
        Sent all model inputs to the appropriate device (GPU on CPU)
        rreturn:
         The inputs are in a dictionary format
        '''
        inputs = {key: (
            value.to(self.device) if (
                    (key not in self.keys_to_ignore_on_gpu) and (value is not None)
            ) else value
        ) for key, value in batch.items()}
        return inputs

    def check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')
        if isinstance(loss, torch.Tensor) and torch.isnan(loss):
            import pdb
            pdb.set_trace()

    def build_writer(self):
        # tensorboard
        if _has_tensorboard:
            msg = f'Initializing summary writer for tensorboard with log_dir={self.opts.output_dir}'
            self.logger.info(msg)
            exp_dir = os.path.join(self.opts.output_dir, f'{self.prefix}_tb_logs')
            self.writer = SummaryWriter(log_dir=exp_dir, comment='Training logs')
            self.writer.add_text("train_arguments", to_json_string(self.opts.__dict__))
        else:
            exp_dir = os.path.join(self.opts.output_dir, f'{self.prefix}_file_logs')
            self.writer = FileWriter(log_dir=exp_dir)

    def build_model_warp(self):
        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.opts.fp16_opt_level)
        # Multi-gpu training (should be after apex fp16 initialization)
        if self.device_num > 1:
            self.model = nn.DataParallel(self.model)

    def train_forward(self, batch):
        '''
        Training forward
        '''
        self.model.train()
        inputs = self.build_batch_inputs(batch)
        if self.use_amp:
            with autocast():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        check_object_type(object=outputs, check_type=dict, name='outputs')
        if self.device_num > 1: outputs['loss'] = outputs['loss'].mean()
        return outputs

    def train_backward(self, loss):
        '''
        Training backward
        '''
        self.check_nan(loss)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

    def train_update(self):
        if self.use_amp:
            # AMP: gradients need unscaling
            self.scaler.unscale_(self.optimizer)
        if self.max_grad_norm is not None and self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                amp.master_params(self.optimizer) if self.use_apex else self.model.parameters(),
                self.max_grad_norm)
        optimizer_was_run = True
        if self.use_amp:
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            scale_after = self.scaler.get_scale()
            optimizer_was_run = scale_before <= scale_after
        else:
            self.optimizer.step()
        if optimizer_was_run: self.scheduler.step()  # Update learning rate schedule
        self.model.zero_grad()  # Reset gradients to zero
        self.global_step += 1

    def train_adv(self, batch):
        if self.opts.adv_type == "fgm":
            self.adv_model.attack()
            adv_outputs = self.train_forward(batch)
            adv_loss = adv_outputs['loss']
            self.train_backward(adv_loss)
            self.adv_model.restore()
        elif self.opts.adv_type == "pgd":
            self.adv_model.backup_grad()
            for t in range(self.opts.adv_number):
                self.adv_model.attack(is_first_attack=(t == 0))
                if t != self.opts.adv_number - 1:
                    self.optimizer.zero_grad()
                else:
                    self.adv_model.restore_grad()
                adv_outputs = self.train_forward(batch)
                adv_loss = adv_outputs['loss']
                self.train_backward(adv_loss)
            self.adv_model.restore()
        elif self.opts.adv_type == "awp":
            # for t in range(self.opts.adv_number):
            #     self.adv_model.attack(is_first_attack=(t == 0))
            #     self.optimizer.zero_grad()
            #     adv_outputs = self.train_forward(batch)
            #     adv_loss = adv_outputs['loss']
            #     self.train_backward(adv_loss)
            # self.adv_model.restore()
            self.adv_model.attack(is_first_attack=True)
            adv_outputs = self.train_forward(batch)
            adv_loss = adv_outputs['loss']
            self.train_backward(adv_loss)
            self.adv_model.restore()

    def train_step(self, step, batch):
        outputs = self.train_forward(batch)
        loss = outputs['loss']
        self.train_backward(loss)
        should_save = False
        should_logging = False
        if self.opts.adv_enable and step >= self.opts.adv_start_steps:
            self.train_adv(batch)
        if (step + 1) % self.gradient_accumulation_steps == 0 or (
                self.steps_in_epoch <= self.gradient_accumulation_steps
                and (step + 1) == self.steps_in_epoch
        ):
            self.train_update()
            should_logging = self.global_step % self.opts.logging_steps == 0
            should_save = self.global_step % self.opts.save_steps == 0
            self.records['loss_meter'].update(loss.item(), n=1)
            self.writer.add_scalar('loss/train_loss', loss.item(), self.global_step)
            if hasattr(self.scheduler, 'get_lr'):
                self.writer.add_scalar('learningRate/train_lr', self.scheduler.get_lr()[0], self.global_step)
            return outputs, should_logging, should_save
        else:
            return None, should_logging, should_save

    # TODO 多机分布式训练
    def train(self, train_data, dev_data=None, resume_path=None, start_epoch=1, state_to_save=dict()):
        train_dataloader = self.build_train_dataloader(train_data)
        num_training_steps = len(train_dataloader) // self.gradient_accumulation_steps * self.num_train_epochs
        self.steps_in_epoch = len(train_dataloader)
        if self.scheduler is None:
            self.scheduler = self.build_lr_scheduler(num_training_steps)
        self.resume_from_checkpoint(resume_path=resume_path)
        self.build_model_warp()
        self.print_summary(len(train_data), num_training_steps)
        self.optimizer.zero_grad()
        seed_everything(self.opts.seed, verbose=False)  # Added here for reproductibility (even between python 2 and 3)
        if self.opts.logging_steps < 0:
            self.opts.logging_steps = len(train_dataloader) // self.gradient_accumulation_steps
            self.opts.logging_steps = max(1, self.opts.logging_steps)
        if self.opts.save_steps < 0:
            self.opts.save_steps = len(train_dataloader) // self.gradient_accumulation_steps
            self.opts.save_steps = max(1, self.opts.save_steps)
        self.build_record_tracker()
        self.reset_metrics()
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training', num_epochs=self.num_train_epochs)
        for epoch in range(start_epoch, int(self.num_train_epochs) + 1):
            pbar.epoch(current_epoch=epoch)
            for step, batch in enumerate(train_dataloader):
                outputs, should_logging, should_save = self.train_step(step, batch)
                if outputs is not None:
                    if self.opts.ema_enable:
                        self.model_ema.update(self.model)
                    pbar.step(step, {'loss': outputs['loss'].item()})
                if (self.opts.logging_steps > 0 and self.global_step > 0) and \
                        should_logging and self.opts.evaluate_during_training:
                    self.evaluate(dev_data)
                    if self.opts.ema_enable and self.model_ema is not None:
                        self.evaluate(dev_data, prefix_metric='ema')
                    if hasattr(self.writer, 'save'):
                        self.writer.save()
                if (self.opts.save_steps > 0 and self.global_step > 0) and should_save:
                    # model checkpoint
                    if self.model_checkpoint:
                        state = self.build_state_object(**state_to_save)
                        if self.opts.evaluate_during_training:
                            if self.model_checkpoint.monitor not in self.records['result']:
                                msg = ("There were expected keys in the eval result: "
                                    f"{', '.join(list(self.records['result'].keys()))}, "
                                    f"but get {self.model_checkpoint.monitor}."
                                    )
                                raise TypeError(msg)
                            self.model_checkpoint.step(
                                state=state,
                                current=self.records['result'][self.model_checkpoint.monitor]
                            )
                        else:
                            self.model_checkpoint.step(
                                state=state,
                                current=None
                            )

            # early_stopping
            if self.early_stopping:
                if self.early_stopping.monitor not in self.records['result']:
                    msg = ("There were expected keys in the eval result: "
                           f"{', '.join(list(self.records['result'].keys()))}, "
                           f"but get {self.early_stopping.monitor}."
                           )
                    raise TypeError(msg)
                self.early_stopping.step(
                    current=self.records['result'][self.early_stopping.monitor])
                if self.early_stopping.stop_training:
                    break
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if self.writer:
            self.writer.close()

    def build_state_object(self, **kwargs):
        '''
        save state object
        '''
        states = {
            'model': self.model.module if hasattr(self.model, "module") else self.model,
            'opts': self.opts,
            'optimizer': self.optimizer,
            'global_step': self.global_step,
        }
        if self.scheduler is not None:
            states['scheduler'] = self.scheduler
        if self.use_amp:
            states['scaler'] = self.scaler
        for key, value in kwargs.items():
            if key not in states:
                states[key] = value
        return states

    def resume_from_checkpoint(self, resume_path=None):
        '''
        Check if continuing training from a checkpoint
        '''
        if resume_path is not None:
            optimizer_path = os.path.join(resume_path, OPTIMIZER_NAME)
            scheduler_path = os.path.join(resume_path, SCHEDULER_NAME)
            state_path = os.path.join(resume_path, TRAINER_STATE_NAME)
            model_path = os.path.join(resume_path, WEIGHTS_NAME)
            scaler_path = os.path.join(resume_path, SCALER_NAME)
            if is_file(optimizer_path):
                self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
            if is_file(scheduler_path):
                self.scheduler.load_state_dict(torch.load(scheduler_path))
            if is_file(state_path):
                state = torch.load(state_path)
                if self.model_checkpoint and hasattr(state, 'best_score'):
                    self.model_checkpoint.best = state['best_score']
                del state
            if is_file(model_path):
                if self.use_amp and is_file(scaler_path):
                    self.scaler.load_state_dict(torch.load(scaler_path))
                load_model(self.model, model_path, device=self.device)

    def print_summary(self, examples, t_total):
        '''
        print training parameters information
        '''
        # self.logger.info("Training/evaluation parameters %s", self.opts)
        self.logger.info("***** Running training %s *****", self.opts.task_name)
        self.logger.info("  Options = %s", self.opts)
        self.logger.info("  Model type = %s", self.opts.model_type)
        self.logger.info("  Num examples = %d", examples)
        self.logger.info("  Num Epochs = %d", self.num_train_epochs)
        self.logger.info("  Instantaneous batch size per GPU = %d", self.opts.per_gpu_train_batch_size)
        self.logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                         self.opts.per_gpu_train_batch_size * self.device_num * self.gradient_accumulation_steps)
        self.logger.info("  Gradient Accumulation steps = %d", self.gradient_accumulation_steps)
        self.logger.info("  Total optimization steps = %d", t_total)
        self.logger.info("  Total Number of Parameters: %d" % sum(p.numel() for p in self.model.parameters()))
        # Calculating total number of trainable params
        self.logger.info("  Total Number of Trainable Parameters: %d " % sum(
            p.numel() for p in self.model.parameters() if p.requires_grad))

    def print_evaluate_result(self):
        '''
        打印evaluation结果,
        '''
        if len(self.records['result']) == 0:
            self.logger.warning("eval result record is empty")
        self.logger.info("***** Evaluating results of %s *****", self.opts.task_name)
        self.logger.info("  global step = %s", self.global_step)
        print_result = []
        for key, value in self.records['result'].items():
            if isinstance(value, (int, float)):
                print_result.insert(0, [key, value])
            elif isinstance(value, pd.DataFrame):
                print_result.append([key, value])
            else:
                print_result.append([key, value])
        for key, value in print_result:
            if isinstance(value, pd.DataFrame):
                self.logger.info(f" %s : \n %s", key, str(round(value, 5)))
            else:
                self.logger.info(f"  %s = %s", key, str(round(value, 5)))
                name = "_".join(key.split("_")[1:]) if "_" in key else key
                self.writer.add_scalar(f"{name}/{key}", value, int(self.global_step / self.opts.logging_steps))

    def save_predict_result(self, data, file_name, save_dir=None):
        '''
        保存预测信息
        '''
        if save_dir is None:
            save_dir = self.opts.output_dir
        elif not os.path.isdir(save_dir):
            save_dir = os.path.join(self.opts.output_dir, save_dir)
        file_path = os.path.join(save_dir, file_name)
        if ".pkl" in file_path:
            save_pickle(file_path=file_path, data=data)
        elif ".json" in file_path:
            json_to_text(file_path=file_path, data=data)
        else:
            raise ValueError("file type: expected one of (.pkl, .json)")

    def evaluate(self, dev_data, prefix_metric=None, save_dir=None, save_result=False, file_name=None):
        '''
        Evaluate the model on a validation set
        '''
        all_batch_list = []
        eval_dataloader = self.build_eval_dataloader(dev_data)
        self.build_record_tracker()
        self.reset_metrics()
        pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
        for step, batch in enumerate(eval_dataloader):
            batch = self.predict_forward(batch)
            if 'loss' in batch and batch['loss'] is not None:
                self.records['loss_meter'].update(batch['loss'], n=1)
            all_batch_list.append(batch)
            pbar.step(step)
        self.records['result']['eval_loss'] = self.records['loss_meter'].avg
        self.update_metrics(all_batch_list, prefix_metric)
        self.print_evaluate_result()
        if save_result:
            if file_name is None: file_name = f"dev_eval_results.pkl"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_metrics(self, all_batch_list, prefix):
        eval_data = self.build_batch_concat(all_batch_list, dim=0)
        prefix = '' if prefix is None else prefix + "_"
        for metric in self.metrics:
            metric.update(preds=eval_data['preds'], target=eval_data['target'])
            value = metric.value()
            if isinstance(value, float):
                self.records['result'][f'{prefix}eval_{metric.name()}'] = value
            elif isinstance(value, dict):
                self.records['result'].update({f"{prefix}eval_{k}": v for k, v in value.items()})
            elif value is None:
                self.logger.info(f"{metric.name()} value is None")
            else:
                msg = "metric value type: expected one of (float, dict,None)"
                raise ValueError(msg)

    def predict(self, test_data, save_result=True, file_name=None, save_dir=None):
        '''
        test数据集预测
        '''
        all_batch_list = []
        test_dataloader = self.build_test_dataloader(test_data)
        pbar = ProgressBar(n_total=len(test_dataloader), desc='Predicting')
        for step, batch in enumerate(test_dataloader):
            batch = self.predict_forward(batch)
            all_batch_list.append(batch)
            pbar.step(step)
        if save_result:
            if file_name is None: file_name = f"test_predict_results.pkl"
            self.save_predict_result(data=all_batch_list, file_name=file_name, save_dir=save_dir)
        return all_batch_list

    def predict_forward(self, batch):
        self.model.eval()
        inputs = self.build_batch_inputs(batch)
        with torch.no_grad():
            outputs = self.model(**inputs)
        if 'loss' in outputs and outputs['loss'] is not None:
            outputs['loss'] = outputs['loss'].mean().detach().item()
        outputs = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value for key, value in
                   outputs.items()}
        batch = {key: value for key, value in dict(batch, **outputs).items() if
                 key not in self.keys_to_ignore_on_result_save}
        return batch

    def build_batch_concat(self, all_batch_list):
        raise NotImplementedError('Method [build_batch_concat] should be implemented.')

import os
import json
import argparse
from pathlib import Path


class Argparser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super(Argparser, self).__init__(**kwargs)

    @classmethod
    def get_training_parser(cls, description='Arguments'):
        parser = cls(description=description, add_help=True)
        parser.arguments_required()
        parser.arguments_common()
        parser.arguments_input_file()
        parser.arguments_dataset()
        parser.arguments_dataloader()
        parser.arguments_pretrained()
        parser.arguments_ema()
        parser.arguments_adv()
        parser.arguments_optimimzer()
        parser.arguments_lr_scheduler()
        parser.arguments_apex()
        parser.arguments_checkpoint()
        parser.arguments_earlystopping()
        return parser

    @classmethod
    def parse_args_from_parser(cls, parser):
        args = parser.parse_args()
        parser.make_experiment_dir(args)
        parser.save_args_to_json(args)
        parser.print_args(args)
        return args

    @classmethod
    def parse_args_from_json(cls, json_file):
        check_file(json_file)
        data = json.loads(Path(json_file).read_text())
        return argparse.Namespace(**data)

    @classmethod
    def get_training_arguments(cls):
        parser = cls.get_training_parser()
        args = cls.parse_args_from_parser(parser)
        return args

    def get_val_argments(self):
        args = Argparser.get_training_arguments()
        return args

    def get_predict_arguments(self):
        args = Argparser.get_training_arguments()
        return args

    def arguments_required(self):
        group = self.add_argument_group(title="required arguments", description="required arguments")
        group.add_argument("--task_name", default=None, type=str, required=True,
                           help="The name of the task to train. ")
        group.add_argument("--output_dir", default=None, type=str, required=True,
                           help="The output directory where the model predictions and checkpoints will be written.")
        group.add_argument("--model_type", default=None, type=str, required=True,
                           help="The name of the model to train.")
        group.add_argument("--data_dir", default=None, type=str, required=True,
                           help="The input data dir. Should contain the training files for task.")

    def arguments_common(self):
        group = self.add_argument_group(title="common arguments", description="common arguments")
        group.add_argument("--seed", type=int, default=42, help="random seed for initialization")
        group.add_argument("--do_train", action="store_true", help="Whether to run training.")
        group.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        group.add_argument("--do_predict", action="store_true", help="Whether to run predict on the test set.")
        group.add_argument("--device_id", type=str, default='0',
                           help='multi-gpu:"0,1,.." or single-gpu:"0" or cpu:"cpu"')
        group.add_argument("--evaluate_during_training", action="store_true",
                           help="Whether to run evaluation during training at each logging step.", )
        group.add_argument('--load_arguments_file', type=str, default=None, help="load args from arguments file")
        group.add_argument("--save_steps", type=int, default=-1,
                           help="Save checkpoint every X updates steps. ``-1`` means that a epoch")
        group.add_argument("--logging_steps", type=int, default=-1,
                           help="Log every X updates steps.``-1`` means that a epoch")
        group.add_argument('--experiment_code', type=str, default='v0', help='experiment code')

    def arguments_input_file(self):
        group = self.add_argument_group(title="input file arguments", description="input file arguments")
        group.add_argument("--train_input_file", default=None, type=str, help="The name of train input file")
        group.add_argument("--eval_input_file", default=None, type=str, help="The name of eval input file")
        group.add_argument("--test_input_file", default=None, type=str, help="The name of test input file")

    def arguments_dataset(self):
        group = self.add_argument_group(title="datasets arguments", description="datasets arguments")
        group.add_argument("--train_max_seq_length", default=128, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument("--eval_max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument("--test_max_seq_length", default=512, type=int,
                           help="The maximum total input sequence length after tokenization. Sequences longer "
                                "than this will be truncated, sequences shorter will be padded.", )
        group.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for training.")
        group.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for evaluation.")
        group.add_argument("--per_gpu_test_batch_size", default=8, type=int,
                           help="Batch size per GPU/CPU for test evaluation.")
        group.add_argument("--overwrite_data_cache", action='store_true',
                           help="Whether to overwrite the cached training and evaluation feature sets")
        group.add_argument("--use_data_cache", action='store_true',
                           help='Whether to load the cached training feature sets')

    def arguments_dataloader(self):
        group = self.add_argument_group(title="dataloader arguments", description="dataloader arguments")
        group.add_argument('--pin_memory', default=False, action='store_true',
                           help='Use pin memory option in data loader')
        group.add_argument("--drop_last", default=False, action='store_true')
        group.add_argument('--num_workers', default=0, type=int, help='Number of data workers')
        group.add_argument("--persistent_workers", default=False, action="store_true", help="")

    def arguments_pretrained(self):
        group = self.add_argument_group(title="pretrained arguments", description="pretrained arguments")
        group.add_argument("--pretrained_model_path", default=None, type=str,
                           help="Path to pre-trained model or shortcut name selected in the list")
        group.add_argument("--pretrained_config_name", default=None, type=str,
                           help="Pretrained config name or path if not the same as model_name")
        group.add_argument("--pretrained_tokenizer_name", default=None, type=str,
                           help="Pretrained tokenizer name or path if not the same as model_name")
        group.add_argument("--do_lower_case", action="store_true",
                           help="Set this flag if you are using an uncased model.")
        group.add_argument("--pretrained_cache_dir", default=None, type=str,
                           help="Where do you want to store the pre-trained models downloaded from s3", )

    def arguments_ema(self):
        group = self.add_argument_group(title='EMA', description='Exponential moving average arguments')
        group.add_argument('--ema_enable', action='store_true', help='Exponential moving average')
        group.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay')
        group.add_argument("--model_ema_force_cpu", action='store_true')

    def arguments_adv(self):
        group = self.add_argument_group(title='Adversarial training', description='Adversarial training arguments')
        group.add_argument('--adv_enable', action='store_true', help='Adversarial training')
        group.add_argument('--adv_start_steps', default=0, type=int, help='the step to start attack')
        group.add_argument('--adv_type', default='fgm', type=str, choices=['fgm', 'pgd', 'awp'])
        group.add_argument('--adv_epsilon', type=float, default=1.0, help='adv epsilon')
        group.add_argument('--adv_name', type=str, default='word_embeddings',
                           help='name for adversarial layer')
        group.add_argument('--adv_number', default=1, type=int, help='the number of attack')
        group.add_argument('--adv_alpha', default=0.3, type=float, help='adv alpha')

    def arguments_optimimzer(self):
        group = self.add_argument_group(title='optimizer', description='Optimizer related arguments')
        group.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
        group.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        group.add_argument("--adam_beta1", default=0.9, type=float, help="Beta1 for AdamW optimizer")
        group.add_argument("--adam_beta2", default=0.999, type=float, help='Beta2 for AdamW optimizer')
        group.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    def arguments_lr_scheduler(self):
        group = self.add_argument_group(title="lr scheduler arguments", description="LR scheduler arguments")
        group.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        group.add_argument("--other_learning_rate", default=0.0, type=float)
        group.add_argument("--base_model_name", default='base_model', type=str, help='The main body of the model.')
        group.add_argument("--num_train_epochs", default=3, type=int, help="Total number of training epochs")
        group.add_argument("--gradient_accumulation_steps", type=int, default=1,
                           help="Number of updates steps to accumulate before performing a backward/update pass.", )
        group.add_argument("--warmup_proportion", default=0.1, type=float,
                           help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
        group.add_argument("--warmup_steps", default=0, type=int,
                           help='Linear warmup over warmup_steps.')
        group.add_argument("--scheduler_type", default='linear', type=str,
                           choices=["linear", 'cosine', 'cosine_with_restarts', 'polynomial', 'constant',
                                    'constant_with_warmup'],
                           help='The scheduler type to use.')

    def arguments_apex(self):
        group = self.add_argument_group(title="apex arguments", description="apex arguments")
        group.add_argument("--fp16", action="store_true",
                           help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit", )
        group.add_argument("--fp16_opt_level", type=str, default="O1",
                           help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html", )
        group.add_argument('--fp16_backend', default='auto', type=str, choices=["auto", "amp", "apex"],
                           help="The backend to be used for mixed precision.")
        group.add_argument("--fp16_full_eval", action='store_true',
                           help="Whether to use full 16-bit precision evaluation instead of 32-bit")

    def arguments_checkpoint(self):
        group = self.add_argument_group(title='model checkpoint', description='model checkpoint arguments')
        group.add_argument("--checkpoint_mode", default='min', type=str, help='model checkpoint mode')
        group.add_argument("--checkpoint_monitor", default='eval_loss', type=str, help='model checkpoint monitor')
        group.add_argument("--checkpoint_save_best", action='store_true', help='Whether to save best model')
        group.add_argument("--checkpoint_verbose", default=1, type=int, help='whether to print checkpoint info')
        group.add_argument("--checkpoint_predict_code", type=str, default=None,
                           help='The version of checkpoint to predict')
        group.add_argument('--eval_all_checkpoints', action="store_true", help="Evaluate all checkpoints starting", )

    def arguments_earlystopping(self):
        group = self.add_argument_group(title='early stopping', description='early stopping arguments')
        group.add_argument("--earlystopping_patience", default=-1, type=int,
                           help='Interval (number of epochs) between checkpoints')
        group.add_argument("--earlystopping_mode", default='min', type=str, help='early stopping mode')
        group.add_argument("--earlystopping_monitor", default='eval_loss', type=str, help='early stopping monitor')
        group.add_argument("--earlystopping_verbose", default=1, type=int, help='whether to print earlystopping info')
        group.add_argument('--earlystopping_save_state_path', default=None, type=str)
        group.add_argument('--earlystopping_load_state_path', default=None, type=str)

    def save_args_to_json(self, args):
        if args.do_train:
            save_arguments_file_name = f"{args.task_name}_{args.model_type}_{args.experiment_code}_opts.json"
            save_arguments_file_path = os.path.join(args.output_dir, save_arguments_file_name)
            if os.path.exists(save_arguments_file_path):
                print(f"[Warning]File {save_arguments_file_path} exist,Overwrite arguments file")
            with open(str(save_arguments_file_path), 'w') as f:
                json.dump(vars(args), f, ensure_ascii=False, indent=4)

    def print_args(self, args):
        print('**********************************')
        print('************ Arguments ***********')
        print('**********************************')
        args_list = sorted(args.__dict__.items(),key=lambda x:x[0])
        msg = ''
        for k,v in args_list:
            msg += f'  {k}: {v}\n'
        print(msg)

    def make_experiment_dir(self, args):
        args.output_dir = os.path.join(args.output_dir, f'{args.task_name}_{args.model_type}_{args.experiment_code}')
        os.makedirs(args.output_dir, exist_ok=True)

import torch
import logging

def prepare_device(device_id):
    """
    setup GPU device if available, move model into configured device
    # 如果输入的是一个list，则默认使用list[0]作为controller
    Example:
        device_id = 'cpu' : cpu
        device_id = '0': cuda:0
        device_id = '0,1' : cuda:0 and cuda:1
     """
    if not isinstance(device_id, str):
        msg = 'device_id should be a str,e.g. multi-gpu:"0,1,.." or single-gpu:"0" or cpu:"cpu"'
        raise TypeError(msg)
    machine_device_num = get_all_available_gpus()
    if machine_device_num == 0 or device_id == 'cpu':
        device_num = 0
        device = torch.device('cpu')
        msg = "Warning: There\'s no GPU available on this machine, training will be performed on CPU."
        logger.warning(msg)
    else:
        logger.info(f"Available GPU\'s: {machine_device_num}")
        device_ids = [int(x) for x in device_id.split(",")]
        device_num = len(device_ids)
        device_type = f"cuda:{device_ids[0]}"
        device = torch.device(device_type)
        if device_num > machine_device_num:
            msg = (f"The number of GPU\'s configured to use is {device_num}, "
                   f"but only {machine_device_num} are available on this machine."
                   )
            logger.warning(msg)
            device_num = machine_device_num
    logger.info("Finally, device: %s, n_gpu: %s", device, device_num)
    return device, device_num


def get_all_available_gpus():
    return torch.cuda.device_count()

