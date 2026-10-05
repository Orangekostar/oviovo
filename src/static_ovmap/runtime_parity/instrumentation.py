"""Exclusive host spans and non-additive asynchronous CUDA service annotations."""

from collections import Counter
from contextlib import contextmanager
import time


STAGES = ("registry_geometry","view_search","rgb_preprocess","recognition","export_bookkeeping","other_sync")


class Stages:
    def __init__(self, *, clock=time.perf_counter):
        self.clock, self.active, self.previous = clock,None,None
        self.seconds = dict.fromkeys(STAGES,0.)
        self.counters = Counter()
        self.events = {"encoder":[],"region_pool_head":[]}

    def _switch(self, name):
        now = self.clock()
        if self.active is not None:
            self.seconds[self.active] += now-self.previous
        self.active,self.previous = name,now

    @contextmanager
    def span(self, name):
        if name not in self.seconds:
            raise ValueError("unknown exclusive recovery stage")
        previous = self.active
        self._switch(name)
        try:
            yield
        finally:
            self._switch(previous)

    @contextmanager
    def cuda_span(self, name, *, device="cuda"):
        if device != "cuda":
            yield
            return
        import torch
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.events[name].append((start,end))

    def finish(self, total):
        if self.active is not None:
            raise ValueError("measured callable returned with an open stage span")
        residual = total-sum(self.seconds.values())
        if residual < -1e-8:
            raise ValueError("exclusive host stage time exceeds synchronized total")
        self.seconds["other_sync"] += max(0.,residual)
        return dict(self.seconds)

    def gpu_seconds(self):
        # Called only after the total end synchronization, never inside region loops.
        return {name:sum(start.elapsed_time(end) for start,end in events)/1000
                for name,events in self.events.items()}
