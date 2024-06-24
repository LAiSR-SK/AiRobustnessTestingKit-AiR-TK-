# (c) 2024 LAiSR-SK
# This code is licensed under the MIT license (see LICENSE.md).
# A simple torch style logger
# (C) Wei YANG 2017
# Copied from https://github.com/bearpaw/pytorch-classification/blob/master/utils/logger.py

from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["Logger", "LoggerMonitor", "savefig"]


def savefig(fname, dpi=None):
    dpi = 150 if dpi is None else dpi
    plt.savefig(fname, dpi=dpi)


def plot_overlap(logger, names=None):
    names = logger.names if names is None else names
    numbers = logger.numbers
    for _, name in enumerate(names):
        x = np.arange(len(numbers[name]))
        plt.plot(x, np.asarray(numbers[name]))
    return [logger.title + "(" + name + ")" for name in names]


class Logger(object):
    """Save training process to log file with simple plot function."""

    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = "" if title is None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, "r")
                name = self.file.readline()
                self.names = name.rstrip().split("\t")
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split("\t")
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(float(numbers[i]))
                self.file.close()
                self.file = open(fpath, "a")
            else:
                self.file = open(fpath, "w")

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write("\t")
            self.numbers[name] = []
        self.file.write("\n")
        self.file.flush()

    def append(self, numbers):
        assert len(self.names) == len(numbers), "Numbers do not match names"
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write("\t")
            self.numbers[self.names[index]].append(num)
        self.file.write("\n")
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names is None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + "(" + name + ")" for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


class LoggerMonitor(object):
    """Load and visualize multiple logs."""

    def __init__(self, paths):
        """paths is a distionary with {name:filepath} pair"""
        self.loggers = []
        for title, path in paths.items():
            logger = Logger(path, title=title, resume=True)
            self.loggers.append(logger)

    def plot(self, names=None):
        plt.figure()
        plt.subplot(121)
        legend_text = []
        for logger in self.loggers:
            legend_text += plot_overlap(logger, names)
        plt.legend(
            legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0
        )
        plt.grid(True)


if __name__ == "__main__":
    # Example
    # logger = Logger('test.txt', resume=True)
    # logger.set_names(['Train loss', 'Valid loss','Test loss'])
    #
    # length = 150
    # t = np.arange(length)
    # train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    # test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
    #
    # for i in range(0, length):
    #     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
    # logger.plot()
    # plt.show()

    # Example: logger monitor
    paths = {
        "resadvnet20": "test1.txt",
        "resadvnet32": "test2.txt",
        "resadvnet44": "test3.txt",
    }

    field = ["Valid Acc."]

    monitor = LoggerMonitor(paths)
    monitor.plot()
    plt.show()
    savefig("test.eps")


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # output: [B, d]
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # pred: [B, len(topk)] -> [len(topk), B]
    # target.reshape(1, -1).expand_as(pred): [len(topk), B]
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    # correct: [len(topk), B]

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res