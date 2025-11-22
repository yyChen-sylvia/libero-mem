import torch
from torchvision import transforms as T
from torchvision.models.optical_flow import raft_small, raft_large


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ToTensor(),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(512, 512)),
        ]
    )
    batch = transforms(batch)
    return batch

class TorchFlow:
    def __init__(self):
        # If you can, run this example on a GPU, it will be a lot faster.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = raft_small(pretrained=True, progress=False).to(self.device)
        self.model = self.model.eval()

    def run(self, img1, img2):
        device = self.device
        img1_batch = preprocess(img1).to(device)
        img2_batch = preprocess(img2).to(device)
        img1_batch = torch.stack([img1_batch])
        img2_batch = torch.stack([img2_batch])

        list_of_flows = self.model(img1_batch.to(device), img2_batch.to(device))

        return list_of_flows
