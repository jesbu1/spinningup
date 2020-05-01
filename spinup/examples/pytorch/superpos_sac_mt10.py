from spinup.utils.run_utils import ExperimentGrid
from spinup import psp_sac_pytorch
import torch
import gym

TASK_HORIZON = 200
NUM_TASKS = 10
PATHS_PER_TASK = 3
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    parser.add_argument('--psp_type', type=str, default='Rand', help='Rand, Ones, Binary, Proposed, Sanity')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=(160, 160, 160, 160, 160))
    args = parser.parse_args()
    hidden_sizes_name = '_'.join([str(num) for num in args.hidden_sizes])
    #eg = ExperimentGrid(name='superpos_sac-MT10_with_bias_%s_context_q_%s' % (args.psp_type, hidden_sizes_name))
    eg = ExperimentGrid(name='TIMETEST')
    eg.add('env_name', 'MT10Helper-v0', '', True)
    eg.add('num_tasks', 10)
    eg.add('batch_size', 128) # This is per task, so real is 128 x 10
    eg.add('psp_type', args.psp_type)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 900)
    eg.add('steps_per_epoch', TASK_HORIZON * PATHS_PER_TASK * NUM_TASKS) 
    eg.add('update_after', TASK_HORIZON * NUM_TASKS) 
    eg.add('lr', [3e-4])
    eg.add('start_steps', 1000)
    #eg.add('update_every', NUM_TASKS * )
    eg.add('num_test_episodes', 10 * NUM_TASKS)
    eg.add('ac_kwargs:hidden_sizes', [tuple(args.hidden_sizes)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.ReLU], '')
    eg.run(psp_sac_pytorch, num_cpu=args.cpu)
#from metaworld.benchmarks import MT10
#
#env_fn = lambda : MTEnv(MT10.get_train_tasks())
#
#ac_kwargs = dict(hidden_sizes=[400,400], activation=torch.nn.ReLU)
#
#logger_kwargs = dict(output_dir='~/spinup/data/', exp_name='SAC_MT10')
#
#sac_pytorch(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=128 * 10, epochs=1000, start_steps=1000, lr=3e-4, logger_kwargs=logger_kwargs)
