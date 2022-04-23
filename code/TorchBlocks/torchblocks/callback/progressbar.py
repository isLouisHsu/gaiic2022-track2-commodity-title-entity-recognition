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
