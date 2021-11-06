
import numpy
import random
import pickle

from src.replay import ReplayBuffer
from src.utils import SumSegmentTree, MinSegmentTree


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.
    
    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        
    """
    
    def __init__(self, obs_dim, size, save_dir, batch_size, alpha, n_step, gamma):
        """Initialization."""
        assert alpha >= 0
        
        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, save_dir, batch_size, n_step, gamma)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        self.save_dir = save_dir
        self.p_chkpt_cnt = 1
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(self, obs, act, rew, next_obs, done):
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)
        
        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        
        return transition

    def sample_batch(self, beta):
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = numpy.array([self._calculate_weight(i, beta) for i in indices])
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)
            
    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size
        
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)
            
        return indices
    
    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)
        
        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight
        
        return weight
    
    def save(self):
        super().save(attribute="priority")

        sum_save_path = self.save_dir / f"sum-tree.pkl"
        with open(sum_save_path, "wb") as fp:
            pickle.dump(self.sum_tree, fp)
        
        min_save_path = self.save_dir / f"min-tree.pkl"
        with open(min_save_path, "wb") as fp:
            pickle.dump(self.min_tree, fp)
        
        misc = dict(max_priority=self.max_priority, tree_ptr=self.tree_ptr, alpha=self.alpha)
        misc_save_path = self.save_dir / f"miscellaneous.pkl"
        with open(misc_save_path, "wb") as fp:
            pickle.dump(misc, fp)
        print(f"[{'priority #' + str(self.p_chkpt_cnt) + ']':>15} Memory Buffer of size {self.tree_ptr} and total capacity {self.max_size} saved to {self.save_dir}")
        self.p_chkpt_cnt += 1

    def load(self, np_chkpt_path, repl_misc_chkpt_path, sum_chkpt_path, min_chkpt_path, misc_chkpt_path):
        # TODO: optimize loader to automate filepaths
        super().load(np_chkpt_path, repl_misc_chkpt_path)

        if not sum_chkpt_path.exists():
            raise ValueError(f"{sum_chkpt_path} does not exist")
        if not min_chkpt_path.exists():
            raise ValueError(f"{min_chkpt_path} does not exist")
        if not misc_chkpt_path.exists():
            raise ValueError(f"{misc_chkpt_path} does not exist")

        with open(sum_chkpt_path, "rb") as fh:
            self.sum_tree = pickle.load(fh)
        
        print(f"Loaded sum tree from {sum_chkpt_path}")

        with open(min_chkpt_path, "rb") as fh:
            self.min_tree = pickle.load(fh)
        
        print(f"Loaded min tree from {min_chkpt_path}")

        with open(misc_chkpt_path, "rb") as fh:
            miscellaneous = pickle.load(fh)
            
            self.max_priority = miscellaneous["max_priority"]
            self.tree_ptr = miscellaneous["tree_ptr"]
            self.alpha = miscellaneous["alpha"]
        
        print(f"Loaded miscellaneous data from {misc_chkpt_path}")

        print(f"Buffer checkpoint reloaded successfully")
