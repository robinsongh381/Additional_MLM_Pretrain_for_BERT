# -*- coding: utf-8 -*-
from __future__ import absolute_import
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

logger = logging.getLogger()


def init_logger(log_dir=None, log_name=None, log_file_level=logging.NOTSET):
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        
    log_file = log_dir+log_name
    if not os.path.exists('/'.join(log_file.split('/')[:-1])):
        os.mkdir('/'.join(log_file.split('/')[:-1]))
        
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger



class TBWriter():
    def __init(self, mode, tb_write_dir):
        self.writer = SummaryWriter('{}/runs'.format(tb_write_dir))
        
    def acc_write(tr_acc, eval_acc, global_step):
        self.writer.add_scalars('acc', {'train': tr_acc, 'val': eval_acc}, global_step)
        
    def loss_write(tr_loss, eval_loss, global_step):
        self.writer.add_scalars('acc', {'train': tr_loss, 'val': eval_loss}, global_step)




class CheckpointManager:
    def __init__(self, model_dir):
        if not isinstance(model_dir, Path):
            model_dir = Path(model_dir)
        self._model_dir = model_dir

    def save_checkpoint(self, state, filename):
        torch.save(state, self._model_dir / filename)

    def load_checkpoint(self, filename):
        state = torch.load(self._model_dir / filename, map_location=torch.device('cpu'))
        return state