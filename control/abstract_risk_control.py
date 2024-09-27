from abc import ABC, abstractmethod


class RiskControl(ABC):
    @abstractmethod
    def set_collector(self):
        pass

    @abstractmethod
    def collect_predictions(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass
