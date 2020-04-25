from spinup.utils.run_utils import ExperimentGrid
from spinup import sac_pytorch
import torch
import gym

TASK_HORIZON = 200
NUM_TASKS = 10

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='sac-MT10')
    eg.add('env_name', 'MT10Helper-v0', '', True)
    eg.add('num_tasks', [10])
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 1000)
    eg.add('steps_per_epoch', TASK_HORIZON * NUM_TASKS) 
    eg.add('lr', [3e-4])
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
