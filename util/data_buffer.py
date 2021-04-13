import numpy as np


class BoolDataBuffer:
    def __init__(self, *args: iter, calc_parity=False):
        self._buffer = np.empty((0,), dtype=np.bool)
        self._index = 0
        for data in args:
            self._buffer = np.append(self._buffer, data).astype(np.bool)

        self._parity = False
        if calc_parity:
            self.calc_parity()

    def next(self, count=1):
        if count == -1:
            ret = self._buffer[self._index:]
            self._index = len(self._buffer)
        else:
            last_index = min(len(self._buffer), self._index + count)
            ret = np.zeros((count,), dtype=np.bool) ^ self._parity
            ret[0: last_index - self._index] = self._buffer[self._index: last_index]
            self._index = last_index
        return ret ^ self._parity

    def add(self, data):
        data_size = len(data)
        if self._index >= data_size:
            self._buffer[self._index - data_size:self._index] = data
            self._index -= data_size
        else:
            self._buffer = np.append(data, self._buffer[self._index:]).astype(np.bool)
            self._index = 0

    def push(self, *args):
        self._buffer = np.concatenate((self._buffer, *args))

    def set_parity(self, parity):
        self._parity = parity

    def get_parity(self):
        return self._parity

    def calc_parity(self):
        self._parity = np.count_nonzero(self._buffer) > self._buffer.size // 2

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __getitem__(self, item):
        return self._buffer[item]

    def clear(self):
        self._buffer = np.array([])
        self._parity = False
