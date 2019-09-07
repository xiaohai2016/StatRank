"""
Basic utility functions and classes
"""
import torch

def get_default_device():
    """Pick a GPU device if available, otherwise, use CPU"""
    if torch.cuda.is_available():
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def to_device(data, device):
    """To move data to a chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(itm, device) for itm in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    """To wrap a DataLoader object and move data to a chosen device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to the device"""
        for data in self.dl:
            yield to_device(data, self.device)

    def __len__(self):
        """Return the number of batches"""
        return len(self.dl)
