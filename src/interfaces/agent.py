from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self,
                 model,
                 device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def invoke(self, query):
        raise NotImplementedError