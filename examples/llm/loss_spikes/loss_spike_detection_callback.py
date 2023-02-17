
import torch
import collections
from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['LossSpikeDetectionCallback']



class MomentSpikeDetector:
    
    def __init__(self,
                 window_moving_average=25,
                 increase_factor=1.5,
                 increase_lookback=500,
                 plateau_min_duration=1000,
                 end_spike_factor=1.15):
          
        self.window_moving_average=window_moving_average
        self.increase_factor=increase_factor
        self.plateau_min_duration=plateau_min_duration
        self.increase_lookback = increase_lookback
        self.fast_moving_average = collections.deque(maxlen=window_moving_average)
        self.intermediate_data_queue = collections.deque(maxlen=increase_lookback-window_moving_average)
        self.slow_moving_average = collections.deque(maxlen=increase_lookback)
        self.end_spike_factor = end_spike_factor
        self.in_spike = False
        self.mva_before_spike = None
        self.spike_batch_idx_start = None


    
    def insert_observation(self, obs, batch_idx):
        if len(self.fast_moving_average) >= self.fast_moving_average.maxlen:
            # move the oldest obs out of the fast moving average into the
            # intermediate data queue
            fast_obs = self.fast_moving_average.popleft()
            
            if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
                # move data from intermediate quque to slow MCVA queue
                intermediate_obs = self.intermediate_data_queue.popleft()
                self.slow_moving_average.append(intermediate_obs)

            self.intermediate_data_queue.append(fast_obs)
        
        self.fast_moving_average.append(obs)
        
        fast_mva = sum(self.fast_moving_average) / len(self.fast_moving_average)
        if not self.in_spike:
            if len(self.slow_moving_average) > self.window_moving_average:
                if self.mva_before_spike is None:
                    slow_mva = sum(self.slow_moving_average) / len(self.slow_moving_average)
                else:
                    slow_mva = self.mva_before_spike
                                    
                
                if fast_mva >= self.increase_factor * slow_mva:
                    self.in_spike = True
                    self.mva_before_spike = slow_mva
                    self.spike_batch_idx_start = batch_idx
        else:
            if batch_idx - self.spike_batch_idx_start > self.plateau_min_duration:
                # kill the run!
                return True
            else:
                if fast_mva <= self.mva_before_spike * self.end_spike_factor:
                    self.in_spike = False
                    self.spike_batch_idx_start = None
        
        return False
  
  
class LossSpikeDetectionCallback(Callback):
    
    def __init__(self,
                 window_moving_average=25,
                 increase_factor=1.5,
                 increase_lookback=500,
                 plateau_min_duration=1000,
                 end_spike_factor=1.15):
          
        self.window_moving_average=window_moving_average
        self.increase_factor=increase_factor
        self.plateau_min_duration=plateau_min_duration
        self.increase_lookback = increase_lookback
        self.fast_moving_average = collections.deque(maxlen=window_moving_average)
        self.intermediate_data_queue = collections.deque(maxlen=increase_lookback-window_moving_average)
        self.slow_moving_average = collections.deque(maxlen=increase_lookback)
        self.end_spike_factor = end_spike_factor
        self.in_spike = False
        self.mva_before_spike = None
        self.spike_batch_idx_start = None

    def divide_chunks(self, l, n):
        # looping till length l
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def get_slow_mva(self):
        min_chunk_mva = max(self.slow_moving_average)
        for chunk in self.divide_chunks(list(self.slow_moving_average), self.window_moving_average):
            chunk_avg = sum(chunk) / len(chunk)
            if chunk_avg < min_chunk_mva:
                min_chunk_mva = chunk_avg
        return min_chunk_mva
    
    def insert_observation(self, obs, batch_idx):
        if len(self.fast_moving_average) >= self.fast_moving_average.maxlen:
            # move the oldest obs out of the fast moving average into the
            # intermediate data queue
            fast_obs = self.fast_moving_average.popleft()
            
            if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
                # move data from intermediate quque to slow MCVA queue
                intermediate_obs = self.intermediate_data_queue.popleft()
                self.slow_moving_average.append(intermediate_obs)

            self.intermediate_data_queue.append(fast_obs)
        
        self.fast_moving_average.append(obs)
        
        fast_mva = sum(self.fast_moving_average) / len(self.fast_moving_average)
        if not self.in_spike:
            if len(self.slow_moving_average) > self.window_moving_average:
                if self.mva_before_spike is None:
                    slow_mva = self.get_slow_mva()
                else:
                    slow_mva = self.mva_before_spike
                                    
                
                if fast_mva >= self.increase_factor * slow_mva:
                    self.in_spike = True
                    self.mva_before_spike = slow_mva
                    self.spike_batch_idx_start = batch_idx
        else:
            if batch_idx - self.spike_batch_idx_start > self.plateau_min_duration:
                # kill the run!
                return True
            else:
                if fast_mva <= self.mva_before_spike * self.end_spike_factor:
                    self.in_spike = False
                    self.spike_batch_idx_start = None
        
        return False
  
    def batch_end(self, state: State, logger: Logger):
        batch_idx = state.timestamp.batch.value
        loss = state.total_loss_dict['loss/train/total']
        is_loss_spike = self.insert_observation(loss, batch_idx)
        if is_loss_spike:
            logger.log_metrics({'loss_spike': batch_idx})
            print(f"Found loss spike starting @ {batch_idx-self.loss_plateau_min_duration}, killing run.")
            state.stop_training()

