# cache datastructure for numerical scalar values
# elements are added at the back using append
# when number of elements exceeds max_cache_size, front elements are deleted
# average of the cache is automatically calculated and stored for convinience
class ScalarCache:
    def __init__(self, max_cache_size:int):
        self.cache = []
        self.max_cache_size = max_cache_size
        self.inv = 1.0 / self.max_cache_size # track this value for quick calculation
        self.m_avg = 0.0

    def clear(self):
        self.cache.clear()
        self.m_avg = 0.0

    def append(self, x):
        self.cache.append(x)
        if len(self.cache) > self.max_cache_size:
            self.m_avg += self.inv*(x - self.cache[0])
            del self.cache[0]
        else:
            self.m_avg = (self.m_avg*(len(self.cache) - 1) + x) / len(self.cache)
    
    def avg(self):
        return self.m_avg