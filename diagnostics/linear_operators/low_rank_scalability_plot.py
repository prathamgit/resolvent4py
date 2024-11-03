import subprocess
import os.path
import matplotlib.pyplot as plt
import numpy as np

fnames = ['scaling_apply.csv', 'scaling_apply_ht.csv', \
          'scaling_apply_mat.csv', 'scaling_apply_ht_mat.csv']

for fname in fnames:
    subprocess.run(['rm', fname]) if os.path.isfile(fname) else None
    subprocess.run(['touch', fname])


nprocs = [1, 2, 5, 10, 20, 30, 40]
for proc in nprocs:
    print(f"Running with {proc} processors")
    subprocess.run(['ibrun', '-n', '%s'%proc, 'python3', \
                    'low_rank_scalability.py'])

plt.figure()
labels = [r'apply()', r'apply_ht()', r'apply_mat()', r'apply_ht_mat()']
for (k, fname) in enumerate(fnames):
    data = np.loadtxt(fname, delimiter=',')[:,-1]
    speedup = data[0]/data
    plt.plot(nprocs, speedup, 'x-', label=labels[k])

plt.plot(nprocs, nprocs, 'k', alpha=0.1)
ax = plt.gca()
ax.set_xlabel(r'Number of MPI processors')
ax.set_ylabel(r'Speedup')
ax.set_title(r'LowRankLinearOperator')
plt.legend()
plt.savefig('scaling_plot.png', dpi=100)

