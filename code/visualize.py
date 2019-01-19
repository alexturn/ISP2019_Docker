import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np

fig = plt.figure()
for T in [1,2,3,4,5]:

    index = np.arange(0, 1.1, 0.1)
    entr_results = np.load('./runs_data/results_entropy_{}.npy'.format(T))
    plt.plot(index, entr_results, label='T = {}'.format(T))

plt.ylabel(r'$H({\varepsilon})$')
plt.xlabel(r'$\varepsilon$')
fig.legend(bbox_to_anchor=(0.3, 0.86))
fig.savefig('../latex/figs/comparison.png')