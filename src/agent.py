
import gc
import gym
import numpy
import torch
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_

from src.core import *
from src.model import RainbowDQN
from src.replay import ReplayBuffer
from src.priority import PrioritizedReplayBuffer


class Daedalous:

    def __init__(self, env, model_save_dir, memory_save_dir, model_checkpoint, mem_checkpoint):

        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_SYNC
        self.gamma = GAMMA

        # Agent checkpoint helper variables
        self.counter = 1
        self.curr_step = 0
        self.save_dir = model_save_dir
        self.chkpt_interval = NUM_OF_STEPS_TO_CHECKPOINT
        self.curr_best_mean_reward = -numpy.inf
        self.ep_rewards = numpy.zeros((NUM_OF_STEPS_TO_CHECKPOINT, 1), dtype=numpy.float)

        # Device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent device\t[{str(self.device):>7}]\n")

        # PER
        # Memory for 1-step Learning
        self.beta = BETA
        self.prior_eps = PRIOR_EPS
        self.priority_replay = PrioritizedReplayBuffer(
            env.observation_space.shape[0], MEM_CAPACITY, memory_save_dir, BATCH_SIZE, alpha=ALPHA, n_step=N_STEP, gamma=GAMMA)

        # Memory for N-step Learning
        self.use_n_step = True if N_STEP > 1 else False
        if self.use_n_step:
            self.n_step = N_STEP
            self.random_replay = ReplayBuffer(
                env.observation_space.shape[0], MEM_CAPACITY, memory_save_dir, BATCH_SIZE, n_step=N_STEP, gamma=GAMMA)

        # Categorical DQN parameters
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.atom_size = N_ATOMS
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size).to(self.device)

        # Networks: online, target
        self.online = RainbowDQN(
            env.action_space.n, self.atom_size, self.support, './data/pretrained/features_layer.pt', './data/pretrained/stats_layer.pt').to(self.device)
        self.target = RainbowDQN(
            env.action_space.n, self.atom_size, self.support, './data/pretrained/features_layer.pt', './data/pretrained/stats_layer.pt').to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.online.train()
        self.target.eval()

        # print(self.online)

        # Optimizer
        self.optimizer = optim.Adam(self.online.parameters(), lr=LEARNING_RATE)

        # Checkpoint loaders
        if model_checkpoint:
            self.load(model_checkpoint / "daedalous.pt")

        if mem_checkpoint:
            self.priority_replay.load(mem_checkpoint / "priority-replay.npz", mem_checkpoint / "priority-replay-misc.pkl",
                                      mem_checkpoint / "sum-tree.pkl", mem_checkpoint / "min-tree.pkl",  mem_checkpoint / "miscellaneous.pkl")
            self.random_replay.load(
                mem_checkpoint / "random-replay.npz", mem_checkpoint / "random-replay-misc.pkl")

    def act(self, state):
        """
        Given a state, choose an action and update value of step.
        Inputs:
        state (numpy.ndarray): A single observation of the current state, dimension is (env.observation_space.shape)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """

        features, stats = state['features'], state['stats']

        # NoisyNet: no epsilon greedy action selection
        features = torch.FloatTensor(features).to(self.device)
        features = features.unsqueeze(0)

        stats = torch.FloatTensor(stats).to(self.device)
        stats = stats.unsqueeze(0)

        action_values = self.online(features, stats)
        action_idx = torch.argmax(action_values, axis=1).item()

        # Increment step
        self.curr_step += 1

        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience replay and priority buffers.
        Inputs:
        state (numpy.ndarray),
        next_state (numpy.ndarray),
        action (int),
        reward (float),
        done (float)
        """

        state = self.encode(state)
        next_state = self.encode(next_state)

        Transition = [state, action, reward, next_state, done]

        # N-step transition
        if self.use_n_step:
            one_step_transition = self.random_replay.store(*Transition)
        # 1-step transition
        else:
            one_step_transition = Transition

        # Add a single step transition
        if one_step_transition:
            self.priority_replay.store(*one_step_transition)

        # Update mean reward array
        self.ep_rewards[(self.curr_step - 1) % NUM_OF_STEPS_TO_CHECKPOINT] = reward

    def recall_priority(self):
        """
        Retrieve a batch of experiences from priority buffer.
        """
        # PER needs beta to calculate weights
        samples = self.priority_replay.sample_batch(self.beta)

        curr_features, curr_stats = self.decode(samples["obs"])
        next_features, next_stats = self.decode(samples["next_obs"])

        curr_features = torch.FloatTensor(curr_features).to(self.device)
        curr_stats = torch.FloatTensor(curr_stats).to(self.device)
        next_features = torch.FloatTensor(next_features).to(self.device)
        next_stats = torch.FloatTensor(next_stats).to(self.device)

        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)

        indices = samples["indices"]
        return curr_features, curr_stats, action, reward, next_features, next_stats, done, weights, indices

    def recall_random(self, indices):
        """
        Retrieve a batch of experiences from random buffer.
        """
        samples = self.random_replay.sample_batch_from_idxs(indices)

        curr_features, curr_stats = self.decode(samples["obs"])
        next_features, next_stats = self.decode(samples["next_obs"])

        curr_features = torch.FloatTensor(curr_features).to(self.device)
        curr_stats = torch.FloatTensor(curr_stats).to(self.device)
        next_features = torch.FloatTensor(next_features).to(self.device)
        next_stats = torch.FloatTensor(next_stats).to(self.device)

        action = torch.LongTensor(samples["acts"]).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        return curr_features, curr_stats, action, reward, next_features, next_stats, done

    def update_model(self, loss):
        """Update the model by gradient descent."""
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

    @torch.no_grad()
    def projection_distribution(self, next_features, next_stats, reward, done, gamma):
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        # Double DQN
        next_action = self.online(next_features, next_stats).argmax(1)
        next_dist = self.target.dist(next_features, next_stats)
        next_dist = next_dist[range(self.batch_size), next_action]

        t_z = reward + (1 - done) * gamma * self.support
        t_z = t_z.clamp(min=self.v_min, max=self.v_max)
        b = (t_z - self.v_min) / delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Fix disappearing probability mass when l = b = u (b is int)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1

        offset = (torch.linspace(0, (self.batch_size - 1) * self.atom_size, self.batch_size).long(
        ).unsqueeze(1).expand(self.batch_size, self.atom_size).to(self.device))

        proj_dist = torch.zeros(next_dist.size(), device=self.device)
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist

    def compute_td_loss(self, curr_features, curr_stats, action, proj_dist):
        """Compute temporal difference loss."""
        dist = self.online.dist(curr_features, curr_stats)
        log_p = torch.log(dist[range(self.batch_size), action])
        td_loss = -torch.sum(proj_dist * log_p, 1)
        return td_loss

    def learn(self, episode):
        if self.curr_step % self.target_update == 0:
            self.sync_Q_target()

        if self.curr_step % self.chkpt_interval == 0:
            # If current model is better than last checkpoint
            if numpy.mean(self.ep_rewards) > self.curr_best_mean_reward:
                # Update best mean reward metric
                self.curr_best_mean_reward = numpy.mean(self.ep_rewards)
                # Update model checkpoint
                self.save()
            # Update memory checkpoint data
            self.priority_replay.save()
            self.random_replay.save()

        if len(self.priority_replay) < self.batch_size:
            return None

        # Sample from memory
        curr_features, curr_stats, action, reward, next_features, next_stats, done, weights, indices = self.recall_priority()

        # Get categorical dqn loss
        proj_dist = self.projection_distribution(next_features, next_stats, reward, done, self.gamma)

        # Get temporal difference
        td_loss = self.compute_td_loss(curr_features, curr_stats, action, proj_dist)

        # PER: importance sampling before average
        loss = torch.mean(td_loss * weights)

        # N-step Learning loss
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            curr_features, curr_stats, action, reward, next_features, next_stats, done = self.recall_random(indices)
            n_step_proj_dist = self.projection_distribution(next_features, next_stats, reward, done, gamma)
            n_step_td_loss = self.compute_td_loss(curr_features, curr_stats, action, n_step_proj_dist)
            td_loss += n_step_td_loss

            # PER: importance sampling before average
            loss = torch.mean(td_loss * weights)

        self.update_model(loss)

        # PER: update priorities
        loss_for_prior = td_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.priority_replay.update_priorities(indices, new_priorities)

        # PER: increase beta
        fraction = min(episode / EPISODES, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

        # NoisyNet: reset noise
        self.online.reset_noise()
        self.target.reset_noise()

        return loss.item()

    def sync_Q_target(self):
        """Hard target model sync with online model."""
        self.target.load_state_dict(self.online.state_dict())

    def save(self):
        save_path = self.save_dir / f"daedalous.pt"

        torch.save(
            dict(
                online=self.online.state_dict(),
                target=self.target.state_dict(),
                optim=self.optimizer.state_dict()
            ),
            save_path)
        print(f"[{'model #' + str(self.counter) + ']':>15} Daedalous saved to {save_path}")

        self.counter += 1

    def load(self, agent_chkpt_path):
        if not agent_chkpt_path.exists():
            raise ValueError(f"{agent_chkpt_path} does not exist")

        ckp = torch.load(agent_chkpt_path, map_location=(self.device))

        online_state_dict = ckp.get('online')
        target_state_dict = ckp.get('target')
        optim_state = ckp.get('optim')

        print(f"Loading model at {agent_chkpt_path}")

        self.online.load_state_dict(online_state_dict)
        self.target.load_state_dict(target_state_dict)
        self.optimizer.load_state_dict(optim_state)

    def encode(self, X):
        """Flattens the dictionary observation.
        """
        return numpy.concatenate((X['features'].flatten(), X['stats'].flatten())).reshape(1, -1)
    
    def decode(self, X):
        """Resets the flattened observation.
        """
        A = numpy.zeros((self.batch_size, 3, 84, 84), dtype=numpy.float32)
        B = numpy.zeros((self.batch_size, 6), dtype=numpy.float32)
        
        for i in range(self.batch_size):
            A[i] = X[i][:-(1 * 6)].reshape(3, 84, 84)
            B[i] = X[i][-(1 * 6):]
        return A, B
