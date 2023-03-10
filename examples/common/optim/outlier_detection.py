
import collections

class OutlierDetector:
    
    def __init__(self,
                 increase_lookback=500,
                 threshold=7.5):
          
        self.intermediate_data_queue = collections.deque(maxlen=increase_lookback)
        self.slow_moving_average = collections.deque(maxlen=increase_lookback)
        self.outlier_threshold = threshold
    
    def insert_observation(self, obs):
        if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
                # move data from intermediate queue to slow MCVA queue
                intermediate_obs = self.intermediate_data_queue.popleft()
                self.slow_moving_average.append(intermediate_obs)

        self.intermediate_data_queue.append(obs)
        
        if len(self.slow_moving_average) > 0:
            slow_mva = sum(self.slow_moving_average) / len(self.slow_moving_average)
            return obs > self.outlier_threshold * slow_mva
        else:
            return False
