from spinup.utils.run_utils import ExperimentGrid
from spinup import sac_pytorch
import torch
import gym

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-goal')
    eg.add('env_name', 'SawyerPush-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 300)
    eg.add('steps_per_epoch', 10 * 150)
    eg.add('lr', [3e-4, 1e-3])
    eg.add('start_steps', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(400,400)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(sac_pytorch, num_cpu=args.cpu)
#from metaworld.benchmarks import MT10
#
#env_fn = lambda : MTEnv(MT10.get_train_tasks())
#
#ac_kwargs = dict(hidden_sizes=[400,400], activation=torch.nn.ReLU)
#
#logger_kwargs = dict(output_dir='~/spinup/data/', exp_name='SAC_MT10')
#
#sac_pytorch(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=128 * 10, epochs=1000, start_steps=1000, lr=3e-4, logger_kwargs=logger_kwargs)
