import torch
import torch.utils.data as data

from os import makedirs
from os.path import exists, join, isfile

SPEC_NAME = "foo"

class FooDataset(data.Dataset):
    def __init__(self, x):
        super().__init__()

        self.x = x
    
    def __len__(self):
        return 10
    
    def __getitem__(self, index):
        return self.x * torch.Tensor([index, 100*index])
    

class BarDataset(data.Dataset):
    def __init__(self, x):
        super().__init__()
        self.parent = FooDataset(x)
        self.location = "./foo"
        if not exists(self.location):
            makedirs(self.location)
    
    def __len__(self):
        return len(self.parent)

    def __getitem__(self, index):
        name = join(self.location, SPEC_NAME) + str(index) + ".pt"
        if isfile(name):
            print(f"loaded '{name}'")
            return torch.load(name)
        else:
            print(f"computed '{name}'")
            ret = self.parent[index]
            torch.save(ret, name)
            return ret