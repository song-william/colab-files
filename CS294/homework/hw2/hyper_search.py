import os
batches = [10000, 30000, 50000]
rates = [0.005, 0.01, 0.02]

template = "python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} -rtg --nn_baseline --exp_name hc_b{}_r{}"

for b in batches:
    for r in rates:
        os.system(template.format(b, r, b, r))
