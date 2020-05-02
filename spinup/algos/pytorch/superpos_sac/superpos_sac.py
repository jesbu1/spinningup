from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import gym
import time
import spinup.algos.pytorch.superpos_sac.core as core
from spinup.utils.logx import EpochLogger
from torch.utils.tensorboard import SummaryWriter


TASK_HORIZON = 200

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k,v in batch.items()}

class MultiTaskReplayBuffer(ReplayBuffer):
    """
    Maintains ``num_tasks`` separate replay buffers of size ``size`` each
    """
    def __init__(self, obs_dim, act_dim, size, num_tasks):
        self.obs_buf = torch.zeros(core.multi_task_combined_shape(num_tasks, size, obs_dim), dtype=torch.float32)
        self.obs2_buf = torch.zeros(core.multi_task_combined_shape(num_tasks, size, obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros(core.multi_task_combined_shape(num_tasks, size, act_dim), dtype=torch.float32)
        self.rew_buf = torch.zeros(core.combined_shape(num_tasks, size), dtype=torch.float32)
        self.done_buf = torch.zeros(core.combined_shape(num_tasks, size), dtype=torch.float32)
        self.ptr = torch.zeros(core.combined_shape(num_tasks), dtype=torch.long)
        self.size, self.max_size = 0, size * num_tasks
        self.num_tasks = num_tasks

    def store(self, obs, act, rew, next_obs, done, task):
        self.obs_buf[task, self.ptr[task]] = torch.from_numpy(obs)
        self.obs2_buf[task, self.ptr[task]] = torch.from_numpy(next_obs)
        self.act_buf[task, self.ptr[task]] = torch.from_numpy(act)
        self.rew_buf[task, self.ptr[task]] = rew
        self.done_buf[task, self.ptr[task]] = done
        self.ptr[task] = (self.ptr[task] + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
    
    def batched_store(self, obs, act, rew, next_obs, done):
        self.obs_buf[:, self.ptr] = torch.from_numpy(obs)
        self.obs2_buf[:, self.ptr] = torch.from_numpy(next_obs)
        self.act_buf[:, self.ptr] = torch.from_numpy(act)
        self.rew_buf[:, self.ptr] = torch.from_numpy(rew)
        self.done_buf[:, self.ptr] = torch.from_numpy(done)
        self.ptr = torch.fmod(self.ptr + 1, self.max_size)
        self.size = min(self.size+self.num_tasks, self.max_size)

    def sample_batch(self, batch_size=32, separate_by_task=False):
        # Returns a (batch_size * num_tasks) x dim dict of tensors
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[:, idxs],
                     obs2=self.obs2_buf[:, idxs],
                     act=self.act_buf[:, idxs],
                     rew=self.rew_buf[:, idxs],
                     done=self.done_buf[:, idxs])
        if separate_by_task:
            return {k: v.cuda() for k,v in batch.items()}
        return {k: v.view(self.num_tasks * batch_size, -1).cuda() for k,v in batch.items()}

def superpos_sac(env_fn, num_tasks, psp_type, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, target_entropy=None, batch_size=128, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=50):
    """
    Soft Actor-Critic (SAC)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.


        num_tasks: The number of tasks for the env in env_fn

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    #from metaworld.benchmarks import MT40
    #test_env = MT40.get_train_tasks()

    # Creating vectorized batch of envs
    #envs = []
    #for i in range(num_tasks):
    #    env = env_fn()
    #    env.set_task(i)
    #    envs.append(env)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(num_tasks, env.observation_space, env.action_space, psp_type, **ac_kwargs).cuda()
    ac_targ = deepcopy(ac).cuda()

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = MultiTaskReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, num_tasks=num_tasks)

    # Learned Log_Alpha
    log_alpha = torch.zeros((num_tasks, 1), requires_grad=True, device="cuda")

    # Alpha Optimizer
    alpha_optimizer = Adam([log_alpha], lr=lr)

    # Target Entropy
    if target_entropy:
        target_entropy = target_entropy
    else:
        target_entropy = -np.prod(env.action_space.shape).item()

    # Count variables (protip: try to get a feel for how different size networks behave!)
    if psp_type == 'Proposed':
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2, ac.context_gen])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d \t context_gen: %d\n'%var_counts)
    else:
        var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
        logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            if psp_type == 'Proposed':
                context_map = ac.context_gen(o)
                a2, logp_a2 = ac.pi(o2, context=context_map['Pi'])
            else:
                a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            #backup = r + gamma * (1 - d) * (q_pi_targ - log_alpha.exp() * logp_a2)
            backup = r + gamma * (1 - d) * (q_pi_targ - logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                      Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        if psp_type == 'Proposed':
            context_map = ac.context_gen(o)
            pi, logp_pi = ac.pi(o, context=context_map['Pi'])
        else:
            pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Compute alpha loss
        loss_alpha = -(log_alpha * (logp_pi + target_entropy).detach()).mean()

        # Entropy-regularized policy loss
        loss_pi = (log_alpha.exp() * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.cpu().detach().numpy())

        return loss_pi, loss_alpha, pi_info

    # Set up optimizers for policy and q-function
    if psp_type == 'Proposed':
        pi_optimizer = Adam(list(ac.pi.parameters()) + list(ac.context_gen.parameters()), lr=lr)
        q_optimizer = Adam(q_params, lr=lr)
    else:
        pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
        q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver({
        'ac': ac, 
        'log_alpha': log_alpha, 
        'optim_pi': pi_optimizer, 
        'optim_q': q_optimizer, 
        'optim_alpha': alpha_optimizer,
        'replay_buffer': replay_buffer})

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi and alpha.
        loss_pi, loss_alpha, pi_info = compute_loss_pi(data)
        alpha_optimizer.zero_grad()
        loss_alpha.backward()
        alpha_optimizer.step()
        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        logger.store(LossAlpha=loss_alpha.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).cuda(), 
                      deterministic)
    
    def get_batched_action(o, deterministic=False):
        return ac.batched_act(torch.as_tensor(np.array(o), dtype=torch.float32).cuda(), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len, success, goalDist, reachDist = test_env.reset(), False, 0, 0, False, None, None
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, info = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
                if 'success' in info:
                    success = info['success'] or success
                if 'goalDist' in  info and info['goalDist'] is not None:
                    goalDist = info['goalDist']
                if 'reachDist' in info and info['reachDist'] is not None:
                    reachDist = info['reachDist']
            if goalDist != None:
                logger.store(TestGoalDist=goalDist)
            if reachDist != None:
                logger.store(TestReachDist=reachDist)
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len, TestSuccess=success)

    # Prepare for interaction with environment
    total_steps = 0
    start_time = time.time()
    writer = SummaryWriter(logger.output_dir)


    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        steps_before = total_steps
        while (total_steps - steps_before) < steps_per_epoch:
            #obs = []
            #ep_rets = []
            #ep_lens = []
            #successes = []
            #for (i, env) in enumerate(envs):
            #    o, ep_ret, ep_len, success = env.reset(task=i), 0, 0, False
            #    obs.append(o)
            #    ep_rets.append(ep_ret)
            #    ep_lens.append(ep_len)
            #    successes.append(success)
            #dones = [False for i in range(num_tasks)] 
            #for step in range(TASK_HORIZON):
            #    # Until start_steps have elapsed, randomly sample actions
            #    # from a uniform distribution for better exploration. Afterwards, 
            #    # use the learned policy. 
            #    if total_steps > start_steps:
            #        action = get_batched_action(obs)
            #    else:
            #        action = [env.action_space.sample() for env in envs]


            #    # Step the env
            #    r_s = []
            #    obs_2 = []
            #    infos = []
            #    for (i, env) in enumerate(envs):
            #        o2, r, d, info = env.step(action[i])
            #        obs_2.append(o2)
            #        r_s.append(r)
            #        infos.append(info)
            #        ep_rets[i] += r
            #        ep_lens[i] += 1
            #        # Ignore the "done" signal if it comes from hitting the time
            #        # horizon (that is, when it's an artificial terminal signal
            #        # that isn't based on the agent's state)
            #        dones[i] = False if ep_lens[i]==max_ep_len else d

            #    # Store experience to replay buffer
            #    replay_buffer.batched_store(
            #        np.array(obs, np.float32).reshape(num_tasks, 1, -1), 
            #        np.array(action, np.float32).reshape(num_tasks, 1, -1), 
            #        np.array(r_s, np.float32).reshape(num_tasks, 1), 
            #        np.array(obs_2, np.float32).reshape(num_tasks, 1, -1), 
            #        np.array(dones, np.float32).reshape(num_tasks, 1),
            #        )

            #    # Super critical, easy to overlook step: make sure to update 
            #    # most recent observation!
            #    obs = obs_2

            #    # End of trajectory handling
            #    for (i, env) in enumerate(envs):
            #        if 'success' in infos[i]:
            #            successes[i] = infos[i]['success'] or successes[i]
            #        if dones[i] or (ep_lens[i] == max_ep_len):
            #            logger.store(EpRet=ep_rets[i], EpLen=ep_lens[i], EpSuccess=successes[i])
            #            obs[i], ep_rets[i], ep_lens[i], successes[i] = env.reset(task=i), 0, 0, False
            #    total_steps += (1 * num_tasks)
                
            for task in range(num_tasks):
                o, ep_ret, ep_len, success = env.reset(task=task), 0, 0, False
                for step in range(TASK_HORIZON):
                    # Until start_steps have elapsed, randomly sample actions
                    # from a uniform distribution for better exploration. Afterwards, 
                    # use the learned policy. 
                    if total_steps > start_steps:
                        a = get_action(o)
                    else:
                        a = env.action_space.sample()

                    # Step the env
                    o2, r, d, info = env.step(a)
                    ep_ret += r
                    ep_len += 1

                    # Ignore the "done" signal if it comes from hitting the time
                    # horizon (that is, when it's an artificial terminal signal
                    # that isn't based on the agent's state)
                    d = False if ep_len==max_ep_len else d

                    # Store experience to replay buffer
                    replay_buffer.store(o, a, r, o2, d, task)

                    # Super critical, easy to overlook step: make sure to update 
                    # most recent observation!
                    o = o2

                    # End of trajectory handling
                    if 'success' in info:
                        success = info['success'] or success
                    if d or (ep_len == max_ep_len):
                        logger.store(EpRet=ep_ret, EpLen=ep_len, EpSuccess=success)
                        o, ep_ret, ep_len, success = env.reset(task=task), 0, 0, False

                    total_steps += 1

                
            # Update handling
            if total_steps >= update_after:
                for j in range(int((num_tasks * TASK_HORIZON)/1)): # Ratio of 1 training step per 1 timesteps
                    batch = replay_buffer.sample_batch(batch_size, separate_by_task=True)
                    update(data=batch)

        # End of epoch handling
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            #logger.save_state({'envs' : envs, 'epoch': epoch, 'total_steps':total_steps}, None)
            logger.save_state({'env' : env, 'epoch': epoch, 'total_steps':total_steps}, None)

        # Test the performance of the deterministic version of the agent.
        #test_agent()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpSuccess', with_min_and_max=True)
        #logger.log_tabular('TestEpRet', with_min_and_max=True)
        #logger.log_tabular('TestEpLen', average_only=True)
        #if 'TestGoalDist' in logger.epoch_dict:
        #    logger.log_tabular('TestGoalDist', with_min_and_max=True)
        #if 'TestReachDist' in logger.epoch_dict:
        #    logger.log_tabular('TestReachDist', with_min_and_max=True)
        #if 'TestSuccess' in logger.epoch_dict:
        #    logger.log_tabular('TestSuccess', average_only=True)
        logger.log_tabular('TotalEnvInteracts', total_steps)
        logger.log_tabular('Q1Vals', with_min_and_max=True)
        logger.log_tabular('Q2Vals', with_min_and_max=True)
        logger.log_tabular('LogPi', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossAlpha', average_only=True)
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        # write context distribution info
        def write_context_info(module_list: nn.ModuleList, name):
            for module in module_list:
                if hasattr(module, "o"):
                    for task in range(num_tasks):
                        writer.add_histogram(str(task) + "/" + name, module.o[task].cpu().detach().cpu().numpy(), global_step=total_steps)
        if epoch % 10 == 0:
            write_context_info(ac.pi.net, "pi")
            write_context_info(ac.q1.q, "q1")
            write_context_info(ac.q2.q, "q2")
        writer.close()



if __name__ == '__main__':
    TASK_HORIZON = 200
    PATHS_PER_TASK = 3
    NUM_TASKS = 10
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='MT10Helper-v0')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=900)
    parser.add_argument('--batch_size', type=int, default=128) # real is 128 x 10
    parser.add_argument('--lr', type=float, default=3e-4) # real is 128 x 10
    parser.add_argument('--epochs', type=int, default=900)
    #eg = ExperimentGrid(name='superpos_sac-MT10_with_bias_%s_context_q_%s' % (args.psp_type, hidden_sizes_name))
    parser.add_argument('--psp_type', type=str, default='Rand')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    exp_name = 'superpos_sac-MT10_with_bias_%s_context_q_%s' % (args.psp_type, str(tuple([args.hid] * args.l)))
    logger_kwargs = setup_logger_kwargs(exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())


    steps_per_epoch = TASK_HORIZON * PATHS_PER_TASK * NUM_TASKS

    superpos_sac(lambda : gym.make(args.env), num_tasks=args.num_tasks, actor_critic=core.MLPActorCritic, psp_type=args.psp_type,
        seed=args.seed, steps_per_epoch=steps_per_epoch, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, update_after=TASK_HORIZON * NUM_TASKS,
        num_test_episodes=NUM_TASKS * 10,
        start_steps=1000, max_ep_len=TASK_HORIZON,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l, activation=torch.nn.ReLU), 
        logger_kwargs=logger_kwargs)