import torch
from torch.nn.functional import pad


class VisualPrompt(torch.nn.Module):
    def __init__(self, image_shape, shape: str, size: int, num: int, clip: bool) -> None:
        super().__init__()
        assert shape in ["full", "pad"]
        assert size < image_shape[-1] / 2
        self.image_shape = image_shape
        self.shape = shape
        self.size = size
        self.num = num
        self.clip = clip
        
        for i in range(self.num):
            setattr(self, f"prompt{i}", torch.nn.parameter.Parameter(data=torch.zeros(image_shape)))
        if self.shape == "full":
            self.register_buffer("mask", torch.ones(self.image_shape))
        else:
            mask = torch.zeros(image_shape[0], image_shape[-2] - 2*size, image_shape[-1] - 2*size)
            mask = pad(mask, (size, size, size, size), value=1)
            self.register_buffer("mask", mask)


    def forward(self, x):
        results = []
        for i in range(self.num):
            prompt = getattr(self, f"prompt{i}")
            if self.clip:
                results.append(torch.clip(x + prompt * self.mask, min=0, max=1))
            else:
                results.append(x + prompt * self.mask)
        return results
