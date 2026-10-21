"""Exit 0 and print the match if run A's final (second-pass) sort equals any of runs B... exactly: same spike times
and the same partition into clusters (cluster numbers may differ). Exit 1 if none matches."""
import sys, glob
import numpy as np, scipy.io as sio
def final(run):
    f = glob.glob(f'{run}/second_pass_*/*/times_*/times_*.mat')
    return sio.loadmat(f[0], variable_names=['cluster_class'])['cluster_class'] if f else None
a = final(sys.argv[1])
for b_run in sys.argv[2:]:
    b = final(b_run)
    if a is None or b is None or a.shape != b.shape or not np.array_equal(a[:, 1], b[:, 1]): continue
    pairs = set(zip(a[:, 0].astype(int), b[:, 0].astype(int)))      # same partition = a one-to-one label map
    if len(pairs) == len(set(a[:, 0])) == len(set(b[:, 0])):
        print(f'identical sort to {b_run}'); sys.exit(0)
sys.exit(1)
