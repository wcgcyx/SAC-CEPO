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
        self.mean_scale = torch.zeros(batch_size, action_dim).fill_(ce_size).to(device)
        self.log_std_scale = torch.zeros(batch_size, action_dim).fill_(ce_size).to(device)
        self.ce_n = ce_n
        self.ce_ne = ce_ne
        self.ce_t = ce_t
        self.ce_size = ce_size
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
        self.policy_net = PolicyNetwork(state_dim, action_dim, log_std_min, log_std_max, self.normal).to(device)
        self.policy_net_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, _ = self.policy_net.predict(state)
        return action.cpu()[0].detach().numpy()

    def get_rollout_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, _ = self.policy_net.forward(state)
        return mean.tanh().cpu()[0].detach().numpy()

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

        # Search for the most optimal policy based on the current value function
        # Initialize start location
        old_mean, old_log_std = self.policy_net.forward(state)
        old_mean = old_mean.detach()
        old_log_std = old_log_std.detach()
        location = torch.cat((old_mean, old_log_std), dim=1)
        # Initialize search scale
        scale = torch.cat((self.mean_scale, self.log_std_scale), dim=1)

        original_shape = self.ce_n, self.batch_size
        compressed_shape = self.ce_n * self.batch_size
        states = state.expand(self.ce_n, self.batch_size, self.state_dim).reshape(compressed_shape, self.state_dim)

        for t in range(self.ce_t):
            samples = self.sample(location, scale, self.ce_n)
            values = self.evaluate(samples, states, original_shape, compressed_shape)
            _, indices = torch.topk(values, k=self.ce_ne, dim=0)
            elite_samples = torch.index_select(samples, dim=0, index=indices.flatten())
            location = elite_samples.mean(dim=0)
            scale = elite_samples.std(dim=0)

        length = location.shape[1] // 2
        target_mean, target_log_std = torch.split(location, length, dim=1)

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
        # Get new action and log-prob
        target_std = target_log_std.exp()
        z = self.normal.sample(sample_shape=target_std.shape)
        action_raw = target_mean + target_std * z
        new_action = torch.tanh(action_raw)
        log_prob = Normal(target_mean, target_std).log_prob(action_raw) - torch.log(1 - new_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        # Learning
        predicted_new_q_value_1 = self.q_net_1.forward(state, new_action)
        predicted_new_q_value_2 = self.q_net_2.forward(state, new_action)
        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)
        target_v_value = predicted_new_q_value - self.alpha * log_prob
        v_loss = nn.MSELoss()(predicted_v_value, target_v_value.detach())
        self.v_net_optimizer.zero_grad()
        v_loss.backward()
        self.v_net_optimizer.step()

        # Training Policy Network
        predicted_mean, predicted_log_std = self.policy_net.forward(state)
        policy_loss = nn.MSELoss()(predicted_mean, target_mean) + nn.MSELoss()(predicted_log_std, target_log_std)
        self.policy_net_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_net_optimizer.step()

        # Updating Target-V Network
        for target_param, param in zip(self.target_v_net.parameters(), self.v_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def sample(self, location, scale, N):
        # Bound normal distribution with 3 stddev
        upper_bound = location + scale * 3
        lower_bound = location - scale * 3
        original_sample = torch.max(torch.min(Normal(location, scale).sample((N,)), upper_bound), lower_bound)
        length = original_sample.shape[2] // 2
        original_sample = torch.split(original_sample, length, dim=2)
        return torch.cat((original_sample[0], original_sample[1].clamp(self.log_std_min, self.log_std_max)), dim=2)

    def evaluate(self, samples, states, original_shape, compressed_shape, epsilon=1e-6):
        length = samples.shape[2] // 2
        mean, log_std = torch.split(samples, length, dim=2)
        std = log_std.exp()
        noise = self.normal.sample(sample_shape=std.shape)
        action_raw = mean + std * noise
        action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        action_reshape = action.reshape(compressed_shape, self.action_dim)
        predicted_new_q_value_1 = self.q_net_1.forward(states, action_reshape)
        predicted_new_q_value_2 = self.q_net_2.forward(states, action_reshape)

        predicted_new_q_value = torch.min(predicted_new_q_value_1, predicted_new_q_value_2)\
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


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, log_std_min, log_std_max, normal):
        super(PolicyNetwork, self).__init__()

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.normal = normal

        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)

        self.mean_layer3 = nn.Linear(256, action_dim)
        self.log_std_layer3 = nn.Linear(256, action_dim)

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        mean = self.mean_layer3(x)
        log_std = self.log_std_layer3(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        z = self.normal.sample(sample_shape=std.shape)
        action_raw = mean + std * z
        action = torch.tanh(action_raw)
        log_prob = Normal(mean, std).log_prob(action_raw) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob
