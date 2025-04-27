import time
from dataclasses import dataclass
from typing import OrderedDict

@dataclass
class Duration:
    time_stamp = []

    def dur_dict(self):
        dd = self.dur()
        d = OrderedDict()
        for i in dd:
            d[i[0]] = i[1]

        return d

    def dur(self):
        total_n = len(self.time_stamp)
        if total_n <= 1:
            return []
        else:
            return [(self.time_stamp[i][0] + "-" + self.time_stamp[i+1][0],
                     self.time_stamp[i+1][1] - self.time_stamp[i][1]) for i in range(total_n-1)]

    def record(self, key=None):
        t = time.perf_counter()
        if key is None:
            key = str(len(self.time_stamp))
        self.time_stamp.append((key, t))

class Profiler:

    def __init__(self, perf_log_file):
        self._perf_log_file = perf_log_file
        self._all_perfs = OrderedDict()

    @property
    def perf_log_file(self):
        return self._perf_log_file

    def add_time_span(self, key) -> Duration:
        self._all_perfs[key] = Duration()
        return self._all_perfs[key]

    def get_time_span(self, key):
        assert key in self._all_perfs.keys()
        return self._all_perfs[key]

    def dur_dict(self):
        ret = OrderedDict()
        for k,v in self._all_perfs.items():
            ret[k] = v.dur_dict()

        return ret


if __name__ == "__main__":
    prof = Profiler("")
    test1 = prof.add_time_span("test1")
    test1.record()
    time.sleep(1)
    test1.record()
    time.sleep(2)
    test1.record("one_record")
    time.sleep(5)
    test1.record()


    from pprint import pprint
    pprint(test1.time_stamp)
    pprint(test1.dur())
    pprint(test1.dur_dict())
    pprint(prof.dur_dict())
