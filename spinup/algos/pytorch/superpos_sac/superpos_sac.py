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
    def __init__(self, obs_dim, act_dim, size, num_tasks):
        self.obs_buf = np.zeros(core.combined_shape(num_tasks, size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(num_tasks, size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(num_tasks, size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(num_tasks, size), dtype=np.float32)
        self.done_buf = np.zeros(core.combined_shape(num_tasks, size), dtype=np.float32)
        self.ptr = np.zeros(core.combined_shape(num_tasks), dtype=np.int32)
        self.size, self.max_size = 0, size * num_tasks

    def store(self, obs, act, rew, next_obs, done, task):
        self.obs_buf[task, self.ptr] = obs
        self.obs2_buf[task, self.ptr] = next_obs
        self.act_buf[task, self.ptr] = act
        self.rew_buf[task, self.ptr] = rew
        self.done_buf[task, self.ptr] = done
        self.ptr[task] = (self.ptr[task] + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[..., idxs],
                     obs2=self.obs2_buf[..., idxs],
                     act=self.act_buf[..., idxs],
                     rew=self.rew_buf[..., idxs],
                     done=self.done_buf[..., idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).cuda() for k,v in batch.items()}



def superpos_sac(env_fn, num_tasks, psp_type, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, target_entropy=None, batch_size=128, start_steps=10000, 
        update_after=None, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=100):
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
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    update_after = num_tasks * steps_per_epoch

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
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Learned Log_Alpha
    log_alpha = torch.zeros((num_tasks, ), requires_grad=True).cuda()

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
            backup = r + gamma * (1 - d) * (q_pi_targ - log_alpha.exp() * logp_a2)

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
    logger.setup_pytorch_saver(ac)

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
        alpha_optimizer.backward()
        alpha_optimizer.step()
        pi_optimizer.zero_grad()
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

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
            for task in range(num_tasks):
                o, ep_ret, ep_len, success = env.reset_with_task(task), 0, False, 0
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
                        o, ep_ret, ep_len = env.reset(), 0, 0

                    total_steps += 1

                
            # Update handling
            if total_steps >= update_after:
                for j in range((num_tasks * TASK_HORIZON)/10): # Ratio of 1 training step per 10 timesteps
                    batch = replay_buffer.sample_batch(batch_size)
                    update(data=batch)

        # End of epoch handling
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs):
            logger.save_state({'env': env}, None)

        # Test the performance of the deterministic version of the agent.
        #test_agent()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('TestEpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('TestEpLen', average_only=True)
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
        logger.log_tabular('LossQ', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()
        # write context distribution info
        def write_context_info(module_list: nn.ModuleList, name):
            for module in module_list:
                if hasattr(module, "o"):
                    for task in range(num_tasks):
                        writer.add_histogram(str(task) + "/" + name, module.o[task].cpu().detach().cpu().numpy(), global_step=t)
        if epoch % 10 == 0:
            write_context_info(ac.pi.net, "pi")
            write_context_info(ac.q1.q, "q1")
            write_context_info(ac.q2.q, "q2")
        writer.close()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--psp_type', type=str, default='Rand')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    superpos_sac(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic, psp_type=args.psp_type,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
