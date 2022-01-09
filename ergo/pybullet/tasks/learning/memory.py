import random

class WorkingMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.contents = []

    def store(self, item):
        if len(self.contents) == self.capacity:
            # forgotten = int(len(self.contents)/2) # middle
            forgotten = random.randrange(len(self.contents)) # random
            self.contents.pop(forgotten)
        self.contents.append(item)

    def clear(self):
        self.contents = []

    def __contains__(self, item):
        return item in self.contents

class LongTermMemory(dict):

    def __init__(self, consolidation_rate):
        super().__init__()
        self.consolidation_rate = consolidation_rate

    def __setitem__(self, key, value):
        if random.random() < self.consolidation_rate:
            super().__setitem__(key, value)

if __name__ == "__main__":

    seq = "abcdefg"

    capacity = 3
    wm = WorkingMemory(capacity)
    for item in seq: wm.store(item)
    assert tuple(wm.contents) == ("a","f","g")

    consolidation_rate = .25
    ltm = LongTermMemory(consolidation_rate)
    for i in range(100):
        ltm[i] = seq[i % len(seq)]
    print(f"len(ltm) = {len(ltm)}")
    assert 10 < len(ltm) < 40
