
import collections

class OutlierDetector:
    
    def __init__(self,
                 threshold: float = 7.5,
                 increase_lookback: int = 500):
          
        self.intermediate_data_queue = collections.deque(maxlen=increase_lookback)
        self.slow_moving_average = collections.deque(maxlen=increase_lookback)
        self.outlier_threshold = threshold
    
    def insert_observation(self, obs: float) -> bool:
        """Inserts obs into the data buffer and returns true if it is an "outlier", defined `outlier_threshold` times larger than
        the windowed moving average from [T-2*`increase_lookback` : T-`increase_lookback`]

        Args:
            obs (float): Numeric observation for the current timestep.

        Returns:
            bool: Indicator of whether the most recent observation was an outlier.
        """
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
    
    def get_slow_mva(self):
        if len(self.slow_moving_average) > 0:
            return sum(self.slow_moving_average) / len(self.slow_moving_average)
        else:
            return 0
