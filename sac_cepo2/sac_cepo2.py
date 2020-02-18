import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from .memory import ReplayMemory, Transition

# Get device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class Agent:

    def __init__(
            self,
            state_dim,
            action_dim,
            alpha,
            lr=3e-4,
            discount=0.99,
            tau=0.005,
            log_std_min=-20,
            log_std_max=2,
            memory_size=1000000,
            batch_size=256,
            ce_n=100,
            ce_ne=5,
            ce_t=10,
            ce_size=0.33):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.alpha = alpha
        self.discount = discount
        self.tau = tau
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size
        self.normal = Normal(torch.tensor(0.0).to(device), torch.tensor(1.0).to(device))
        self.ce_n = ce_n
        self.ce_ne = ce_ne
        self.ce_t = ce_t
        self.ce_size = ce_size
        self.ce_loc = torch.zeros(batch_size, action_dim).to(device)
        self.ce_scale = torch.zeros(batch_size, action_dim).fill_(ce_size).to(device)
        # SAC has five networks
        # Two Q Networks, One V Network, One Target-V Network and One Policy Network
        self.q_net_1 = QNetwork(state_dim, action_dim).to(device)
        self.q_net_1_optimizer = optim.Adam(self.q_net_1.parameters(), lr=lr)
        self.q_net_2 = QNetwork(state_dim, action_dim).to(device)
        self.q_net_2_optimizer = optim.Adam(self.q_net_2.parameters(), lr=lr)
        self.v_net = VNetwork(state_dim).to(device)
        self.v_net_optimizer = optim.Adam(self.v_net.parameters(), lr=lr)
        self.target_v_net = VNetwork(state_dim).to(device)
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(param.data)
        self.policy_mean_net = PolicyNetworkMean(state_dim, action_dim).to(device)
        self.policy_mean_net_optimizer = optim.Adam(self.policy_mean_net.parameters(), lr=lr)
        self.policy_std_net = PolicyNetworkStd(state_dim, action_dim, log_std_min, log_std_max).to(device)
        self.policy_std_net_optimizer = optim.Adam(self.policy_std_net.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std = self.policy_mean_net.forward(state), self.policy_std_net.forward(state)
        z = self.normal.sample(sample_shape=std.shape)
        action = (mean + std * z).tanh()
        return action.cpu()[0].detach().numpy()

    def get_rollout_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean = self.policy_mean_net.forward(state)
        action = mean.tanh()
        return action.cpu()[0].detach().numpy()

    def store_transition(self, state, action, reward, next_state, end):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = torch.FloatTensor(action).unsqueeze(0).to(device)
        reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
        end = torch.FloatTensor([end]).unsqueeze(0).to(device)
        self.memory.push(state, action, reward, next_state, end)

    def learn(self):
        # Sample experiences
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state = torch.cat(batch.state)
        action = torch.cat(batch.action)
        reward = torch.cat(batch.reward)
        next_state = torch.cat(batch.next_state)
        end = torch.cat(batch.end)

        # Training Q Networks
        predicted_q_value_1 = self.q_net_1.forward(state, action)
        predicted_q_value_2 = self.q_net_2.forward(state, action)
        predicted_v_target = self.target_v_net.forward(next_state)
        target_q_value = reward + (1 - end) * self.discount * predicted_v_target
        q_loss_1 = nn.MSELoss()(predicted_q_value_1, target_q_value.detach())
        q_loss_2 = nn.MSELoss()(predicted_q_value_2, target_q_value.detach())
        self.q_net_1_optimizer.zero_grad()
        q_loss_1.backward()
        self.q_net_1_optimizer.step()
        self.q_net_2_optimizer.zero_grad()
        q_loss_2.backward()
        self.q_net_2_optimizer.step()

        # Training V Network
        predicted_v_value = self.v_net.forward(state)
        mean, std = self.policy_mean_net.forward(state), self.policy_std_net.forward(state)
        z = self.normal.sample(sample_shape=std.shape)
        action_raw = mean + std * z
        new_action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - new_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        predicted_new_q_value_1 = self.q_net_1.forward(state, new_action)
        predicted_new_q_value_2 = self.q_net_2.forward(state, new_action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        target_v_value = predicted_new_q_value - self.alpha * log_prob
        v_loss = nn.MSELoss()(predicted_v_value, target_v_value.detach())
        self.v_net_optimizer.zero_grad()
        v_loss.backward()
        self.v_net_optimizer.step()

        # Search for the most optimal mean based on the current std and value function
        location = mean.detach()
        static_std = std.detach()
        # Initialise search scale
        scale = self.ce_scale

        original_shape = self.ce_n, self.batch_size
        compressed_shape = self.ce_n * self.batch_size
        states = state.expand(self.ce_n, self.batch_size, self.state_dim).reshape(compressed_shape, self.state_dim)

        for t in range(self.ce_t):
            samples = self.sample(location, scale)
            values = self.evaluate(samples, static_std, states, original_shape, compressed_shape)
            _, indices = torch.topk(values, k=self.ce_ne, dim=0)
            elite_samples = torch.index_select(samples, dim=0, index=indices.flatten())
            location = elite_samples.mean(dim=0)
            scale = elite_samples.std(dim=0)
        target_mean = location

        # Training Policy Mean Network
        predicted_mean = self.policy_mean_net.forward(state)
        policy_mean_loss = nn.MSELoss()(predicted_mean, target_mean)
        self.policy_mean_net_optimizer.zero_grad()
        policy_mean_loss.backward()
        self.policy_mean_net_optimizer.step()

        # Training Policy Std Network
        new_mean = self.policy_mean_net.forward(state).detach()
        std = self.policy_std_net.forward(state)
        z = self.normal.sample(sample_shape=std.shape)
        action_raw = new_mean + std * z
        new_action = torch.tanh(action_raw)
        log_prob = Normal(new_mean, std).log_prob(action_raw) - torch.log(1 - new_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        predicted_new_q_value_1 = self.q_net_1.forward(state, new_action)
        predicted_new_q_value_2 = self.q_net_2.forward(state, new_action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        policy_std_loss = (self.alpha * log_prob - predicted_new_q_value).mean()
        self.policy_std_net_optimizer.zero_grad()
        policy_std_loss.backward()
        self.policy_std_net_optimizer.step()

        # Updating Target-V Network
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def sample(self, location, scale):
        # Bound normal distribution with 3 stddev
        upper_bound = location + scale * 3
        lower_bound = location - scale * 3
        return torch.max(torch.min(Normal(location, scale).sample((self.ce_n,)), upper_bound), lower_bound)

    def evaluate(self, samples, static_std, states, original_shape, compressed_shape, epsilon=1e-6):
        mean = samples
        noise = self.normal.sample(sample_shape=static_std.shape)
        action_raw = mean + static_std * noise
        action = action_raw.tanh()
        log_prob = Normal(mean, static_std).log_prob(action_raw) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        action_reshape = action.reshape(compressed_shape, self.action_dim)
        predicted_new_q_value_1 = self.q_net_1.forward(states, action_reshape)
        predicted_new_q_value_2 = self.q_net_2.forward(states, action_reshape)

        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2) \
            .reshape(original_shape[0], original_shape[1], 1)
        loss = (predicted_new_q_value - self.alpha * log_prob).mean(dim=1)
        return loss.detach()


class QNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim + action_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class VNetwork(nn.Module):

    def __init__(self, state_dim):
        super(VNetwork, self).__init__()

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 1)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class PolicyNetworkMean(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(PolicyNetworkMean, self).__init__()

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class PolicyNetworkStd(nn.Module):

    def __init__(self, state_dim, action_dim, log_std_min, log_std_max):
        super(PolicyNetworkStd, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))
        x = self.layer3(x).clamp(self.log_std_min, self.log_std_max)
        return x.exp()
