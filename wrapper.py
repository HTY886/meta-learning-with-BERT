# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
# ==============================================================================

"""Meta-learners for Omniglot experiment."""
import random
import os
from abc import abstractmethod
import torch
from torch import nn
from torch import optim
from tqdm import tqdm
from leap import Leap
from leap.utils import clone_state_dict
from utils_meta import Res, AggRes

from tensorboardX import SummaryWriter


class BaseWrapper(object):

    """Generic training wrapper.

    Arguments:
        args: main args
        model (nn.Module): BERTQA.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
    """

    def __init__(self, args, model, optimizer_cls, optimizer_kwargs):
        self.args = args
        self.model = model
        
        self.optimizer_cls = optim.SGD if optimizer_cls.lower() == 'sgd' else optim.Adam
        self.optimizer_kwargs = optimizer_kwargs
        self.amp = None
        if self.args.local_rank in [-1, 0]:
            self.tb_writer = SummaryWriter(os.path.join(args.output_dir,'log'))

    def __call__(self, tasks, global_step, meta_train=True):
        return self.run_tasks(tasks, global_step, meta_train=meta_train)

    @abstractmethod
    def _partial_meta_update(self, loss, final):
        """Meta-model specific meta update rule.

        Arguments:
            loss (nn.Tensor): loss value for given mini-batch.
            final (bool): whether iteration is the final training step.
        """
        NotImplementedError('Implement in meta-learner class wrapper.')

    @abstractmethod
    def _final_meta_update(self):
        """Meta-model specific meta update rule."""
        NotImplementedError('Implement in meta-learner class wrapper.')
    
    def set_amp(self, amp):
        self.amp = amp

    def zero_grad(self):
        self.model.zero_grad()

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)
    
    def run_tasks(self, tasks, global_step, meta_train):
        """Train on a mini-batch tasks and evaluate test performance.

        Arguments:
            tasks (list, torch.utils.data.DataLoader): list of task-specific dataloaders.
            meta_train (bool): whether current run in during meta-training.
        """
        results = []
        for task in tqdm(tasks, desc="tasks"):
            #task.dataset.train()
            trainres, global_step = self.run_task(task, train=True, meta_train=meta_train, global_step=global_step)
            #task.dataset.eval()???
            valres, _ = self.run_task(task, train=False, meta_train=False, global_step=0)
            results.append((trainres, valres))
        ##
        #results = AggRes(results)

        # Meta gradient step
        if meta_train:
            self._final_meta_update()

        return results, global_step

    def run_task(self, task, train, meta_train, global_step=None):
        """Run model on a given task.

        Arguments:
            task (torch.utils.data.DataLoader): task-specific dataloaders.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        optimizer = None
        if train:
            self.model.train()
            '''
                freeze BertEncoder
            '''
            '''
            p_list = []
            for n, param in self.model.named_parameters():
                if 'output' in n:
                    p_list.append(param)
            ts_params = nn.ParameterList(p_list)
            
            optimizer = self.optimizer_cls(
                ts_params, **self.optimizer_kwargs)
            
            '''
            optimizer = self.optimizer_cls(
                self.model.parameters(), **self.optimizer_kwargs)
            
        else:
            self.model.eval()

        return self.run_batches(task, optimizer, train=train, meta_train=meta_train, global_step=global_step)

    def run_batches(self, batches, optimizer, train=False, meta_train=False, global_step=None):
        """Iterate over task-specific batches.

        Arguments:
            batches (torch.utils.data.DataLoader): task-specific dataloaders.
            optimizer (torch.nn.optim): optimizer instance if training is True.
            train (bool): whether to train on task.
            meta_train (bool): whether to meta-train on task.
        """
        device = next(self.model.parameters()).device

        res = Res()
        N = len(batches)
        batch_iterator = batches
        tr_loss, logging_loss = 0.0, 0.0
        for n, batch in enumerate(batch_iterator):
            #print("Occupied GPU memory: {}".format(torch.cuda.memory_allocated(device=device)))
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
            
            outputs = self.model(**inputs)
            # EVALUATION
            loss = outputs[0]

            res.log(loss=loss.item())

            # TRAINING #
            if not train:
                continue

            if self.args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            
            
            if self.args.fp16:
                with self.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.amp.master_params(optimizer), self.args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            
            final = (n+1) == N
            
            if meta_train:
                self._partial_meta_update(loss, final)
            
            tr_loss += loss.item()
            if (n + 1) % self.args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()     
                if self.args.local_rank in [-1, 0] and self.args.logging_steps > 0 and (global_step+n+1) % self.args.logging_steps == 0:
                    # Log metrics
                    self.tb_writer.add_scalar('loss', (tr_loss - logging_loss)/self.args.logging_steps, global_step+n+1)
                    logging_loss = tr_loss
            torch.cuda.empty_cache()
            
            if final:
                break
        

        res.aggregate()
        return res, global_step+N
    def close_writer(self):
        if self.args.local_rank in [-1, 0]:
            self.tb_writer.close()

class NoWrapper(BaseWrapper):

    """Wrapper for baseline without any meta-learning.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        criterion (func): loss criterion to use.
    """
    def __init__(self, args, model, optimizer_cls, optimizer_kwargs):
        super(NoWrapper, self).__init__(args, model, optimizer_cls, optimizer_kwargs)
        self._original = clone_state_dict(model.state_dict(keep_vars=True))

    def __call__(self, tasks, meta_train=False):
        return super(NoWrapper, self).__call__(tasks, global_step,meta_train=False)

    def run_task(self, *args, **kwargs):
        out = super(NoWrapper, self).run_task(*args, **kwargs)
        self.model.load_state_dict(self._original)
        return out

    def _partial_meta_update(self, loss, final):
        pass

    def _final_meta_update(self):
        pass


class _FOWrapper(BaseWrapper):

    """Base wrapper for First-order MAML and Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = None

    def __init__(self, args, model, optimizer_cls, meta_optimizer_cls, optimizer_kwargs,
                 meta_optimizer_kwargs, optimizer=None, scheduler=None):
        super(_FOWrapper, self).__init__(args, model, optimizer_cls, optimizer_kwargs)
        self.meta_optimizer_cls = optim.SGD if meta_optimizer_cls.lower() == 'sgd' else optim.Adam
        self.meta_optimizer_kwargs = meta_optimizer_kwargs

        self._counter = 0
        self._updates = None
        self._original = clone_state_dict(self.model.state_dict(keep_vars=True))

        params = [p for p in self._original.values() if getattr(p, 'requires_grad', False)]
        if optimizer:
            self.meta_optimizer = optimizer
        else:
            self.meta_optimizer = self.meta_optimizer_cls(params, **meta_optimizer_kwargs)
        self.scheduler = scheduler
    def run_task(self, task, train, meta_train, global_step):
        if meta_train:
            self._counter += 1
        if train:
            self.model.load_state_dict(self._original)
        return super(_FOWrapper, self).run_task(task, train, meta_train, global_step)

    def _partial_meta_update(self, loss, final):
        if not final:
            return

        if self._updates is None:
            self._updates = {}
            for n, p in self._original.items():
                if not getattr(p, 'requires_grad', False):
                    continue

                if p.size():
                    self._updates[n] = p.new(*p.size()).zero_()
                else:
                    self._updates[n] = p.clone().zero_()

        for n, p in self.model.state_dict(keep_vars=True).items():
            if n not in self._updates:
                continue

            if self._all_grads is True:
                self._updates[n].add_(p.data)
            else:
                if p.requires_grad and p.grad is not None:
                    self._updates[n].add_(p.grad.data)

    def _final_meta_update(self):
        for n, p in self._updates.items():
            p.data.div_(self._counter)

        for n, p in self._original.items():
            if n not in self._updates:
                continue

            if self._all_grads:
                p.grad = p.data - self._updates[n].data
            else:
                p.grad = self._updates[n]

        self.meta_optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.meta_optimizer.zero_grad()
        self._counter = 0
        self._updates = None


class ReptileWrapper(_FOWrapper):

    """Wrapper for Reptile.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = True

    def __init__(self, *args, **kwargs):
        super(ReptileWrapper, self).__init__(*args, **kwargs)


class FOMAMLWrapper(_FOWrapper):
    """Wrapper for FOMAML.

    Arguments:
        model (nn.Module): classifier.
        optimizer_cls: optimizer class.
        meta_optimizer_cls: meta optimizer class.
        optimizer_kwargs (dict): kwargs to pass to optimizer upon construction.
        meta_optimizer_kwargs (dict): kwargs to pass to meta optimizer upon construction.
        criterion (func): loss criterion to use.
    """

    _all_grads = False

    def __init__(self, *args, **kwargs):
        super(FOMAMLWrapper, self).__init__(*args, **kwargs)

