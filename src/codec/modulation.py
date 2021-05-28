from abc import ABC, abstractmethod


class Modulation(ABC):
    @abstractmethod
    def value2vector(self, value, flatten=False):
        pass

    @abstractmethod
    def vector2value(self, vector):
        pass

    @abstractmethod
    def get_vector_size(self):
        pass