from abc import ABC, abstractmethod

class LLM(ABC):
    def __init__(self,
                 model,
                 device):
        super().__init__()
        self.model = model
        self.device = device

    @abstractmethod
    def invoke(self, query):
        raise NotImplementedError