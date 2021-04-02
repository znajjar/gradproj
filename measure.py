import time


class Measure:
    def __init__(self, function=None, label="function", return_time=False):
        self.function = function
        self.label = label
        self._return_time = return_time
        self.total_time = 0
        if not function:
            self._last_check_point = time.time()

    def __call__(self, *args, **kwargs):
        start_time = time.time()
        returned = self.function(*args)
        execution_time = time.time() - start_time
        self.total_time += execution_time
        if self._return_time:
            return execution_time, returned
        else:
            print(f'{self.label} execution time:{round(execution_time, 3)} seconds')
            return returned

    def check(self):
        now = time.time()
        dif = now - self._last_check_point
        # print(now - self._last_check_point)
        self.total_time += dif
        self._last_check_point = now
        return dif

    def __str__(self):
        return self.check().__round__(5).__str__() + ' seconds'
