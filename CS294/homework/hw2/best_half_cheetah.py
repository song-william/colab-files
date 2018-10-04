import os

batch = 50000
rate = 0.02

cmds = [
"python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} --exp_name hc_b{}_r{}",
"python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} -rtg --exp_name hc_b{}_r{}",
"python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} --nn_baseline --exp_name hc_b{}_r{}",
'python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.95 -n 100 -e 3 -l 2 -s 32 -b {} -lr {} -rtg --nn_baseline --exp_name hc_b{}_r{}',
]

[os.system(cmd.format(batch, rate, batch, rate)) for cmd in cmds]
