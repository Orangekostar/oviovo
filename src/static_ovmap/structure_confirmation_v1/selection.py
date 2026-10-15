"""One DEV-only choice per method from seed17, without a scientific gain gate."""
from functools import cmp_to_key

def choose_step(rows):
    rows=[r for r in rows if all(r.get(k) is not None for k in ('A','M','C'))]
    if not rows:raise ValueError('No defined DEV checkpoint available')
    def compare(a,b):
        for key in ('A','M','C'):
            delta=a[key]-b[key]
            if abs(delta)>1e-10:return -1 if delta>0 else 1
        return (a['step']>b['step'])-(a['step']<b['step'])
    return sorted(rows,key=cmp_to_key(compare))[0]
