import time
from dataclasses import dataclass
from typing import OrderedDict
import json

@dataclass
class Duration:
    key : str
    time_stamp : list
    skip : int

    def new_iter(self):
        self.time_stamp.append([])

    def dur_dict(self):
        dd = self.dur()
        d = OrderedDict()
        for i in dd:
            d[i[0]] = i[1]

        return d

    def dur_avg(self):
        it = len(self.time_stamp)
        ret = OrderedDict()
        for i in range(self.skip, it):
            durs = self.dur(i)
            for t in durs:
                k = t[0]
                v = t[1]
                if k in ret.keys():
                    ret[k] += v
                else:
                    ret[k] = v

        # do averaging
        for k in ret.keys():
            ret[k] = ret[k]/(it-self.skip)

        return ret

    def dur(self, index):
        assert index < len(self.time_stamp)
        total_n = len(self.time_stamp[index])
        if total_n <= 1:
            return []
        else:
            return [(self.time_stamp[index][i][0] + "-" + self.time_stamp[index][i+1][0],
                     self.time_stamp[index][i+1][1] - self.time_stamp[index][i][1]) for i in range(total_n-1)]

    def record(self, key=None):
        t = time.perf_counter()
        if key is None:
            key = str(len(self.time_stamp[-1]))
        self.time_stamp[-1].append((key, t))

class Profiler:

    def __init__(self, perf_log_file, skip=1):
        self._perf_log_file = perf_log_file
        self._all_perfs = OrderedDict()
        self._skip = skip

    @property
    def perf_log_file(self):
        return self._perf_log_file

    def add_time_span(self, key) -> Duration:
        l = []
        new_dur = Duration(key=key, time_stamp=l, skip=self._skip)
        self._all_perfs[key] = new_dur 
        return self._all_perfs[key]

    def get_time_span(self, key):
        assert key in self._all_perfs.keys()
        return self._all_perfs[key]

    def dur_dict(self):
        ret = OrderedDict()
        for k,v in self._all_perfs.items():
            ret[k] = v.dur_avg()

        return ret

    def dump_results(self):
        with open(self.perf_log_file, "w+") as fp:
            json.dump(self.dur_dict(), fp, indent=4)

if __name__ == "__main__":
    prof = Profiler("")
    test1 = prof.add_time_span("test1")
    test2 = prof.add_time_span("test2")
    print(test2)

    for i in range(10):
        test1.new_iter()
        test2.new_iter()
        test1.record()
        time.sleep(0.005)
        test2.record()
        test1.record()
        time.sleep(0.0007)
        test1.record("one_record")
        time.sleep(0.000001)
        test1.record()
        test2.record()


    from pprint import pprint
    pprint(test1.time_stamp)
    pprint(test1.dur(0))
    pprint(test1.dur_avg())
    pprint(test2.dur_avg())

    pprint(prof.dur_dict())
