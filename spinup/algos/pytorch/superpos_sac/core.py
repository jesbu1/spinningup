import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

class HashLinear(nn.Linear):
    """
    Base Class
    """
    def __init__(self, n_in, n_out, bias=True):
        super(HashLinear, self).__init__(n_in, n_out, bias)

    def forward(self, x, task_index):
        if self.alpha_mode:
            return self.alpha_forward(x, task_index)
        o = self.o[task_index]
        m = x * o
        r = F.linear(m, self.weight, self.bias)
        return r

    def alpha_forward(self, x, alpha):
        o = torch.matmul(self.o, alpha)
        m = torch.matmul(x, o)
        r = F.linear(m, self.weight, self.bias)
        return r

class OnesHashLinear(HashLinear):
    """
    All ones initialization
    """
    def __init__(self, n_in, n_out, num_tasks, learn_key=True, alpha_mode=False):
        super(OnesHashLinear, self).__init__(n_in, n_out)
        self.alpha_mode = alpha_mode
        if alpha_mode:
            o = torch.ones((n_in, n_in, num_tasks))
        else:
            o = torch.ones(size=(num_tasks, n_in))

        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

class RandHashLinear(HashLinear):
    """
    Normal Random Initialization
    """
    def __init__(self, n_in, n_out, num_tasks, learn_key=True, alpha_mode=False):
        super(RandHashLinear, self).__init__(n_in, n_out)
        self.alpha_mode = alpha_mode
        if alpha_mode:
            o = torch.randn(size=(n_in, n_in, num_tasks))
        else:
            o = torch.randn(size=(num_tasks, n_in))

        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

class BinaryHashLinear(HashLinear):
    """
    Random {+1, -1} initialization
    """
    def __init__(self, n_in, n_out, num_tasks, learn_key=True, alpha_mode=False):
        super(BinaryHashLinear, self).__init__(n_in, n_out)
        self.alpha_mode = alpha_mode
        if alpha_mode:
            rand_01 = np.random.binomial(p=.5, n=1, size=(n_in, n_in, num_tasks)).astype(np.float32)
        else:
            rand_01 = np.random.binomial(p=.5, n=1, size=(num_tasks, n_in)).astype(np.float32)
        o = torch.from_numpy(rand_01*2 - 1)

        self.o = nn.Parameter(o)
        if not learn_key:
            self.o.requires_grad = False

class SanityCheckLinear(HashLinear):
    """
    Just for sanity checking, almost equivalent to independent networks
    """
    def __init__(self, n_in, n_out, num_tasks, learn_key=False):
        super(SanityCheckLinear, self).__init__(n_in, n_out)
        o = np.zeros(shape=(num_tasks, n_in))
        mask = np.concatenate((np.ones(int(n_in//num_tasks)), np.zeros(int(n_in - n_in//num_tasks)))).astype(int)
        for task in range(num_tasks):
            o[task] = np.where(mask, 1, 0)
            mask = np.roll(mask, int(n_in//num_tasks))
        o = torch.from_numpy(o).float()
        self.o = nn.Parameter(o)
        self.o.requires_grad = False

class ProposedContextLinear(nn.Linear):
    """
    This takes in a context vector proposed by another network
    """
    def __init__(self, n_in, n_out, num_tasks=None): #numtasks not used
        super(ProposedContextLinear, self).__init__(n_in, n_out)
        w = nn.init.xavier_normal_(torch.empty(n_in, n_out))
        self.w = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(n_out))
        self.o = None

    def forward(self, x, context):
        m = x * context
        r = torch.mm(m, self.w) + self.bias
        return r

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def mlp_psp(sizes, activation, num_tasks, psp_type, output_activation=nn.Identity):
    layers = []
    linear_layer = select_linear_layer(psp_type)
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        if j == 0:
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        else:
            layers += [linear_layer(sizes[j], sizes[j+1], num_tasks), act()]
    return nn.ModuleList(layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def select_linear_layer(psp_type: str):
    if psp_type == 'Ones':
        linear_layer = OnesHashLinear
    elif psp_type == 'Rand':
        linear_layer = RandHashLinear
    elif psp_type == 'Binary':
        linear_layer = BinaryHashLinear
    elif psp_type == 'Proposed':
        linear_layer = ProposedContextLinear
    elif psp_type == 'Sanity':
        linear_layer = SanityCheckLinear
    return linear_layer


LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, num_tasks, obs_dim, act_dim, hidden_sizes, activation, act_limit, psp_type):
        super().__init__()
        self.activation = activation
        self.net = mlp_psp([obs_dim] + list(hidden_sizes), activation, num_tasks, psp_type, activation)
        linear_layer = select_linear_layer(psp_type)
        self.mu_layer = linear_layer(hidden_sizes[-1], act_dim, num_tasks)
        self.log_std_layer = linear_layer(hidden_sizes[-1], act_dim, num_tasks)
        self.act_limit = act_limit
        self.num_tasks = num_tasks

    def forward(self, obs, deterministic=False, with_logprob=True, context=None):
        which_task = torch.argmax(obs[..., -self.num_tasks:], dim=-1).long()
        net_out = obs[..., :-self.num_tasks]
        layer_counter = 0
        for layer in self.net:
            if not hasattr(layer, "o"):
                net_out = layer(net_out)
            elif context is None:
                net_out = layer(net_out, which_task)
            else:
                net_out = layer(net_out, context[layer_counter])
                layer_counter += 1
        if context is None:
            mu = self.mu_layer(net_out, which_task)
            log_std = self.log_std_layer(net_out, which_task)
        else:
            mu = self.mu_layer(net_out, context[-2])
            log_std = self.log_std_layer(net_out, context[-1])
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPQFunction(nn.Module):

    def __init__(self, num_tasks, obs_dim, act_dim, hidden_sizes, activation, psp_type):
        super().__init__()
        #self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
        psp_type = 'Rand' if psp_type == 'Proposed' else psp_type
        self.q = mlp_psp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, num_tasks, psp_type)
        self.num_tasks = num_tasks
        self.activation = activation

    def forward(self, obs, act):
        which_task = torch.argmax(obs[..., -self.num_tasks:], dim=-1).long()
        q = torch.cat([obs[..., :-self.num_tasks], act], dim=-1)
        for layer in self.q:
            if not hasattr(layer, "o"):
                q = layer(q)
            else:
                q = layer(q, which_task)
        #q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1) # Critical to ensure q has right shape.

class ContextGenerator(nn.Module):

    def __init__(self, num_tasks, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        proposal_layers = {}
        #proposal_layers['Q1'] = [obs_dim + act_dim] + hidden_sizes + [1]
        #proposal_layers['Q2'] = proposal_layers['Q1']
        #proposal_layers['Pi'] = [obs_dim] + hidden_sizes + [hidden_sizes[-1]] * 2 #mu and log_std
        proposal_layers['Pi'] = [obs_dim] + list(hidden_sizes) + [hidden_sizes[-1]] * 2 #mu and log_std
        #proposal_layers = [obs_dim + act_dim] * 2 # Q functions
        #proposal_layers.extend([obs_dim]) # Pi Function
        #proposal_layers.extend(hidden_sizes * 3) # 3 sets of hidden layers
        #proposal_layers.extend([1] * 2) # Q function action dimensions
        #proposal_layers.extend([hidden_sizes[-1]] * 2) # Mu and Log_Std size
        self.proposal_layers = proposal_layers
        self.num_tasks = num_tasks
        all_layers = [item for item in proposal_layers.values()]
        self.proposal_network = mlp([num_tasks] + list(hidden_sizes) + [np.sum(all_layers)], activation)

    def forward(self, obs):
        obs = obs[..., -self.num_tasks:]
        context_layers = self.proposal_network(obs)
        context_map = {'Pi': []}
        prev_shape = 0
        for shape in self.proposal_layers['Pi']:
            context_map['Pi'].append(context_layers[..., prev_shape:prev_shape + shape])
            prev_shape += shape
        return context_map


class MLPActorCritic(nn.Module):

    def __init__(self, num_tasks, observation_space, action_space, psp_type, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()
        self.num_tasks = num_tasks
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(num_tasks, obs_dim - num_tasks, act_dim, hidden_sizes, activation, act_limit, psp_type)
        self.q1 = MLPQFunction(num_tasks, obs_dim - num_tasks, act_dim, hidden_sizes, activation, psp_type)
        self.q2 = MLPQFunction(num_tasks, obs_dim - num_tasks, act_dim, hidden_sizes, activation, psp_type)

        # build context proposal function
        self.psp_type = psp_type
        if psp_type == 'Proposed':
            self.context_gen = ContextGenerator(num_tasks, obs_dim - num_tasks, act_dim, hidden_sizes, activation)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            if self.psp_type == 'Proposed':
                context_list = self.context_gen(obs)
                a, _ = self.pi(obs.unsqueeze(0), deterministic, False, context=context_list['Pi'])
            else:
                a, _ = self.pi(obs.unsqueeze(0), deterministic, False)
            return a.squeeze(0).cpu().detach().numpy()
