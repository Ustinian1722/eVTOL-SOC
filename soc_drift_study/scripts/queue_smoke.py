import numpy as np
import pandas as pd

def simulate_queue(events, delay, length):
    active = -1
    results = []
    for t in range(length):
        arrivals = [e for e in events if e + delay + 1 == t]
        if arrivals: active = arrivals[-1]
        results.append(active)
    return results
