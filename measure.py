import time


class Measure:
    precision = 5

    def __init__(self, function=None, label="function", return_time=False, print_time=False):
        self.function = function
        self.label = label
        self._return_time = return_time
        self._total_time = 0
        self._calls_counter = 0
        self._print_time = print_time
        self._last_check_point = time.time()

    def __call__(self, *args, **kwargs):
        self._calls_counter += 1
        start_time = time.time()
        returned = self.function(*args, **kwargs)
        execution_time = time.time() - start_time
        self._total_time += execution_time

        if self._print_time:
            print(f'{self.label} execution time:{round(execution_time, self.precision)} seconds')

        if self._return_time:
            return execution_time, returned
        else:
            return returned

    def check(self):
        now = time.time()
        dif = now - self._last_check_point
        self._total_time += dif
        self._last_check_point = now
        return dif

    def get_average(self):
        return self._total_time / self._calls_counter

    def get_total(self):
        return self._total_time

    def __str__(self):
        return self.check().__round__(self.precision).__str__() + ' seconds'
