import numpy as np
import pickle

class ReplayBuffer:
    def __init__(self, capacity, seq_len):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = []
        self.position = 0
        self.full = False

    # when the buffer is full. start overwrite from the beginning
    def add(self, frame, action, reward, next_frame, done, truncated=False):
        transition = {
            "frame": frame,
            "action": action,
            "reward": reward,
            "next_frame": next_frame,
            "done": done,
            "truncated": truncated,
        }

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity
        if len(self.buffer) == self.capacity:
            self.full = True

    # randomly select batches from the buffer
    def sample_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return batch

    # check whether [start, end] crosses the current write pointer in a full circular buffer
    # if it does, that sequence contains overwritten / invalid ordering
    def _crosses_write_pointer(self, start, end):
        if not self.full:
            return False

        # normal case
        if start <= end:
            return start <= self.position <= end
        # wrapped case
        else:
            return self.position >= start or self.position <= end

    """
    A valid sequence:
    1. has seq_len contiguous steps
    2. does not cross the write pointer if buffer is full
    3. does not contain done/truncated before the last element
    """
    def _is_valid_sequence_start(self, start_idx):
        n = len(self.buffer)
        if n < self.seq_len:
            return False

        if not self.full:
            if start_idx + self.seq_len > n:
                return False
            indices = list(range(start_idx, start_idx + self.seq_len))
        else:
            end_idx = (start_idx + self.seq_len - 1) % self.capacity

            if self._crosses_write_pointer(start_idx, end_idx):
                return False

            indices = [(start_idx + i) % self.capacity for i in range(self.seq_len)]

        # episode boundary check:
        # done/truncated can appear at the last step, but not in the middle
        for i in indices[:-1]:
            if self.buffer[i]["done"] or self.buffer[i].get("truncated", False):
                return False

        return True

    def sample_sequence_batch(self, batch_size):
        n = len(self.buffer)
        if n < self.seq_len:
            raise ValueError("Not enough transitions to sample a sequence batch.")

        # select valid start indices
        if not self.full:
            candidate_starts = list(range(0, n - self.seq_len + 1))
        else:
            candidate_starts = list(range(0, self.capacity))

        valid_starts = [s for s in candidate_starts if self._is_valid_sequence_start(s)]

        if len(valid_starts) < batch_size:
            raise ValueError(
                f"Not enough valid sequences. requested={batch_size}, available={len(valid_starts)}"
            )

        start_indices = np.random.choice(valid_starts, batch_size, replace=False)

        batch_sequences = []
        for start in start_indices:
            if not self.full:
                seq = self.buffer[start:start + self.seq_len]
            else: # collect a contiguous sequence with wrap-around in the circular buffer
                idxs = [(start + i) % self.capacity for i in range(self.seq_len)]
                seq = [self.buffer[i] for i in idxs]
            batch_sequences.append(seq)

        frames = np.stack([np.stack([step["frame"] for step in seq], axis=0) for seq in batch_sequences], axis=0)
        actions = np.stack([np.stack([step["action"] for step in seq], axis=0) for seq in batch_sequences], axis=0)
        rewards = np.stack([np.array([step["reward"] for step in seq], dtype=np.float32) for seq in batch_sequences], axis=0)
        next_frames = np.stack([np.stack([step["next_frame"] for step in seq], axis=0) for seq in batch_sequences], axis=0)
        dones = np.stack([np.array([step["done"] for step in seq], dtype=np.float32) for seq in batch_sequences], axis=0)
        truncateds = np.stack([np.array([step.get("truncated", False) for step in seq], dtype=np.float32) for seq in batch_sequences], axis=0)

        return {
            "frames": frames,  # (B, T, ...)
            "actions": actions,  # (B, T, ...)
            "rewards": rewards,  # (B, T)
            "next_frames": next_frames,  # (B, T, ...)
            "dones": dones,  # (B, T)
            "truncateds": truncateds,  # (B, T)
        }


    def __len__(self):
        return len(self.buffer)

    # save and load sequence buffer
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "buffer": self.buffer,
                "position": self.position,
                "capacity": self.capacity,
                "seq_len": self.seq_len,
                "full": self.full,
            }, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            self.buffer = data["buffer"]
            self.position = data["position"]
            self.capacity = data["capacity"]
            self.seq_len = data["seq_len"]
            self.full = data["full"]
        else:
            # backward compatibility
            self.buffer = data
            self.position = len(self.buffer) % self.capacity
            self.full = len(self.buffer) == self.capacity