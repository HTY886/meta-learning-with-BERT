import os
from os.path import join
import numpy as np

def convert_arg(arg):
    """Convert string to type"""
    # pylint: disable=broad-except
    if arg.lower() == 'none':
        arg = None
    elif arg.lower() == 'false':
        arg = False
    elif arg.lower() == 'true':
        arg = True
    elif '.' in arg:
        try:
            arg = float(arg)
        except Exception:
            pass
    else:
        try:
            arg = int(arg)
        except Exception:
            pass
    return arg


def build_kwargs(args):
    """Build a kwargs dict from a list of key-value pairs"""
    kwargs = {}

    if not args:
        return kwargs

    assert len(args) % 2 == 0, "argument list %r does not appear to have key, value pairs" % args

    while args:
        k = args.pop(0)
        v = args.pop(0)
        if ':' in v:
            v = tuple(convert_arg(a) for a in v.split(':'))
        else:
            v = convert_arg(v)
        kwargs[str(k)] = v

    return kwargs

def compute_auc(x):
    """Compute AUC (composite trapezoidal rule)"""
    T = len(x)
    v = 0
    for i in range(1, T):
        v += ((x[i] - x[i-1]) / 2 + x[i-1]) / T
    return v

class Res:

    """Results container
    Attributes:
        losses (list): list of losses over batch iterator
        meta_loss (float): auc over losses
        loss (float): mean loss over losses. Call ``aggregate`` to compute.
    """

    def __init__(self):
        self.losses = []
        self.meta_loss = 0
        self.loss = 0

    def log(self, loss):
        """Log loss"""
        self.losses.append(loss)

    def aggregate(self):
        """Compute aggregate statistics"""
        self.losses = np.array(self.losses)

        self.loss = self.losses.mean()
        self.meta_loss = compute_auc(self.losses)

class AggRes:

    """Results aggregation container
    Aggregates results over a mini-batch of tasks
    """

    def __init__(self, results):
        self.train_res, self.val_res = zip(*results)
        self.aggregate_train()
        self.aggregate_val()

    def aggregate_train(self):
        """Aggregate train results"""
        (self.train_meta_loss,
         self.train_loss,
         self.train_losses,) = self.aggregate(self.train_res)

    def aggregate_val(self):
        """Aggregate val results"""
        (self.val_meta_loss,
         self.val_loss,
         self.val_losses,) = self.aggregate(self.val_res)

    @staticmethod
    def aggregate(results):
        """Aggregate losses and accs across Res instances"""
        agg_losses = np.stack([res.losses for res in results], axis=1)
        mean_loss = agg_losses.mean()
        mean_losses = agg_losses.mean(axis=1)
        mean_meta_loss = compute_auc(mean_losses)

        return mean_meta_loss, mean_loss, mean_losses
