import time
from dataclasses import dataclass
from typing import OrderedDict, List
import json
import pandas as pd

@dataclass
class Pair:
    start : float
    end : float

    def st(self):
        self.start = time.perf_counter()

    def ed(self):
        self.end = time.perf_counter()

    def dur(self):
        return self.end - self.start

@dataclass
class Record:
    time_stamp_pair : List[OrderedDict[str, Pair]]
    skip : int
    iter_n : int 
    cur_iter : int = 0

    def init(self):
        self.time_stamp_pair = [OrderedDict() for _ in range(self.iter_n)]
        self.cur_iter = 0

    def new_iter(self):
        self.cur_iter += 1

    def new(self, key):
        d = self.time_stamp_pair[self.cur_iter]
        if key in d.keys():
            raise ValueError(f"Key {key} already exists in the current iteration.")

        self.time_stamp_pair[self.cur_iter][key] = Pair(start=0, end=0)
        return self.time_stamp_pair[self.cur_iter][key]

    def report(self):
        ret = OrderedDict()
        first = self.time_stamp_pair[self.skip]
        for k,v in first.items():
            ret[k] = v.dur()

        for i in range(self.skip+1, self.iter_n):
            d = self.time_stamp_pair[i]
            for k, v in d.items():
                ret[k] += v.dur()

        for k in ret.keys():
            ret[k] /= (self.iter_n - self.skip)

        return ret

@dataclass
class Duration:
    key : str
    time_stamp : list
    time_stamp_pair : List[OrderedDict[str, Pair]]
    skip : int
    iter_n : int = 10
    cur_iter : int = 0

    def new_iter(self):
        self.cur_iter += 1

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

    def record_pair(self, key):
        self.time_stamp_pair[-1][key] = Pair(start=0, end=0)
        return self.time_stamp_pair[-1][key]


class Profiler:

    def __init__(self, skip=1, itern=10, perf_log_file ="perf_log.json"):
        self._perf_log_file = perf_log_file
        self._all_perfs = OrderedDict()
        self._skip = skip
        self._iter_n = itern

    @property
    def iter_n(self):
        return self._iter_n

    @property
    def perf_log_file(self):
        return self._perf_log_file

    def add_time_span(self, key) -> Duration:
        d = []
        new_dur = Record(time_stamp_pair=d, skip=self._skip, iter_n=self._iter_n)
        new_dur.init()
        self._all_perfs[key] = new_dur 
        return self._all_perfs[key]

    def get_time_span(self, key):
        assert key in self._all_perfs.keys()
        return self._all_perfs[key]

    def dump_results(self):
        with open(self.perf_log_file, "w+") as fp:
            json.dump(self.report(), fp, indent=4)

    def report(self):
        ret = OrderedDict()
        for k, v in self._all_perfs.items():
            ret[k] = v.report()
        return ret

def test_pair():
    l1 = []
    l2 = []
    d = Duration("test", l1, l2, 0)
    d.new_iter()
    p1 = d.record_pair("test1")
    p2 = d.record_pair("aatest2")
    p1.st()
    time.sleep(0.0001)
    p1.ed()
    p2.st()
    time.sleep(0.0001)
    p2.ed()
    print(d.time_stamp_pair)


if __name__ == "__main__":
    #test_pair()

    prof = Profiler()
    test1 = prof.add_time_span("t1")
    test2 = prof.add_time_span("t2")

    for i in range(10):
        t1 = test1.new("test1")
        t2 = test1.new("test2")
        t3 = test2.new("test3")

        t1.st()
        time.sleep(0.005)
        t1.ed()

        t2.st()
        time.sleep(0.0007)
        t2.ed()

        t3.st()
        time.sleep(0.100001)
        t3.ed()

        test1.new_iter()
        test2.new_iter()

    from pprint import pprint
    pprint(prof.report())
    
    