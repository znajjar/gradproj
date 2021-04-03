import numpy as np


class DataBuffer:
    def __init__(self, dtype, *args: iter):
        self._buffer = np.empty((0,), dtype=dtype)
        self._index = 0
        for data in args:
            self._buffer = np.append(self._buffer, data)

    def next(self, count=1):
        if count == -1:
            ret = self._buffer[self._index:]
            self._index = len(self._buffer)
        else:
            last_index = min(len(self._buffer), self._index + count)
            ret = np.zeros((count,), dtype=np.bool)
            ret[0: last_index - self._index] = self._buffer[self._index: last_index]
            self._index = last_index
        return ret

    def add(self, data):
        self._buffer = np.append(data, self._buffer[self._index:])
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()
