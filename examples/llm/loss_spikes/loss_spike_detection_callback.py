
import torch
import collections
from composer.core import Callback, State
from composer.loggers import Logger

__all__ = ['LossSpikeDetectionCallback']


class LossSpikeDetectionCallback(Callback):
    
    def __init__(self,
                 window_moving_average=25,
                 loss_increase_factor=2,
                 loss_increase_lookback=1000,
                 loss_plateau_min_duration=200,
                 end_loss_spike_factor=1.10):
          
        self.window_moving_average=window_moving_average
        self.loss_increase_factor=loss_increase_factor
        self.loss_plateau_min_duration=loss_plateau_min_duration
        self.loss_increase_lookback = loss_increase_lookback
        self.fast_moving_average = collections.deque(maxlen=window_moving_average)
        self.intermediate_data_queue = collections.deque(maxlen=loss_increase_lookback-window_moving_average)
        self.slow_moving_average = collections.deque(maxlen=loss_increase_lookback)
        self.end_loss_spike_factor = end_loss_spike_factor
        self.in_loss_spike = False
        self.loss_mva_before_spike = None
        self.loss_spike_batch_idx_start = None


    
    def insert_observation(self, loss_obs, batch_idx):
        if len(self.fast_moving_average) >= self.fast_moving_average.maxlen:
            # move the oldest obs out of the fast moving average into the
            # intermediate data queue
            fast_obs = self.fast_moving_average.popleft()
            
            if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
                # move data from intermediate queue to slow MCVA queue
                intermediate_obs = self.intermediate_data_queue.popleft()
                self.slow_moving_average.append(intermediate_obs)

            self.intermediate_data_queue.append(fast_obs)
        
        self.fast_moving_average.append(loss_obs)
        
        fast_mva = sum(self.fast_moving_average) / len(self.fast_moving_average)
        if not self.in_loss_spike:
            if len(self.slow_moving_average) > self.window_moving_average:
                if self.loss_mva_before_spike is None:
                    slow_mva = sum(self.slow_moving_average) / len(self.slow_moving_average)
                else:
                    slow_mva = self.loss_mva_before_spike
                                    
                
                if fast_mva >= self.loss_increase_factor * slow_mva:
                    self.in_loss_spike = True
                    self.loss_mva_before_spike = slow_mva
                    self.loss_spike_batch_idx_start = batch_idx
        else:
            if batch_idx - self.loss_spike_batch_idx_start > self.loss_plateau_min_duration:
                # kill the run!
                return True
            else:
                if fast_mva <= self.loss_mva_before_spike * self.end_loss_spike_factor:
                    self.in_loss_spike = False
                    self.loss_spike_batch_idx_start = None
        
        return False
            

    def batch_end(self, state: State, logger: Logger):
        batch_idx = state.timestamp.batch.value
        loss = state.total_loss_dict['loss/train/total']
        is_loss_spike = self.insert_observation(loss, batch_idx)
        if is_loss_spike:
            logger.log_metrics({'loss_spike': batch_idx})
            print(f"Found loss spike starting @ {batch_idx-self.loss_plateau_min_duration}, killing run.")
            state.stop_training()

