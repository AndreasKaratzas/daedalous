
import numpy
import pickle

from collections import deque


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim, size, save_dir, batch_size, n_step, gamma):
        self.obs_buf = numpy.zeros([size, obs_dim], dtype=numpy.float32)
        self.next_obs_buf = numpy.zeros([size, obs_dim], dtype=numpy.float32)
        self.acts_buf = numpy.zeros([size], dtype=numpy.float32)
        self.rews_buf = numpy.zeros([size], dtype=numpy.float32)
        self.done_buf = numpy.zeros(size, dtype=numpy.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        self.save_dir = save_dir
        self.r_chkpt_cnt = 1
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, obs, act, rew, next_obs, done):
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(self.n_step_buffer, self.gamma)
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self):
        idxs = numpy.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )
    
    def sample_batch_from_idxs(self, idxs):
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )
    
    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def save(self, attribute="random"):
        save_path = self.save_dir / f"{attribute}-replay.npz"
        numpy.savez_compressed(save_path, obs=self.obs_buf, next_obs=self.next_obs_buf, acts=self.acts_buf, rews=self.rews_buf, done=self.done_buf)
        
        misc = dict(ptr=self.ptr, max_size=self.max_size)
        repl_misc_save_path = self.save_dir / f"{attribute}-replay-misc.pkl"
        with open(repl_misc_save_path, "wb") as fp:
            pickle.dump(misc, fp)

        print(f"[{' ' + attribute[0].upper() + '. replay #' + str(self.r_chkpt_cnt) + ']':>15} Random Memory Buffer of size {self.ptr} and total capacity {self.max_size} saved to {save_path}")

        self.r_chkpt_cnt += 1

    def load(self, np_chkpt_path, misc_repl_path):
        if not np_chkpt_path.exists():
            raise ValueError(f"{np_chkpt_path} does not exist")

        if not misc_repl_path.exists():
            raise ValueError(f"{misc_repl_path} does not exist")
        
        chkpt = numpy.load(np_chkpt_path)

        self.obs_buf = chkpt['obs']
        self.acts_buf = chkpt['acts']
        self.rews_buf = chkpt['rews']
        self.done_buf = chkpt['done']
        self.next_obs_buf = chkpt['next_obs']

        with open(misc_repl_path, "rb") as fh:
            misc = pickle.load(fh)

            self.ptr = misc["ptr"]
            self.max_size = misc["max_size"] if misc["max_size"] > self.max_size else self.max_size
        
        print(f"Loading replay of size {self.ptr} and total capacity {self.max_size} from {np_chkpt_path}")

    def __len__(self):
        return self.size
