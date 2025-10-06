
import numpy as np

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval
        self.overfull_count = 0
        self.dictionary = {}

    def count(self):
        return len(self.dictionary)

    def full(self):
        return len(self.dictionary) >= self.size

    def get_index(self, obj, read_only=False):
        d = self.dictionary
        if obj in d:
            return d[obj]
        elif read_only:
            return None
        elif len(d) >= self.size:
            self.overfull_count += 1
            return hash(obj) % self.size
        else:
            new_index = len(d)
            d[obj] = new_index
            return new_index

def tiles(iht, num_tilings, floats, ints=None, read_only=False):
    """
    Returns a list of active tile indices for the given state.
    """
    if ints is None:
        ints = []
    
    qfloats = [np.floor(f * num_tilings) for f in floats]
    tiles_list = []
    
    for tiling in range(num_tilings):
        tilingX2 = tiling * 2
        coords = [tiling]
        
        for i, q in enumerate(qfloats):
            if q >= 0:
                coords.append((q + i + tilingX2) // num_tilings)
            else:
                coords.append((q - i - tilingX2) // num_tilings)
        
        coords.extend(ints)
        tiles_list.append(iht.get_index(tuple(coords), read_only))
        
    return tiles_list
