from typing import Callable

from lightning.pytorch.loops.fetchers import _DataFetcher

from llm_training.lms.model_provider import ModelProvider
from llm_training.models.base_model.base_model import BaseModel

ModelType = ModelProvider | BaseModel | Callable[[], BaseModel]


def get_model(model_or_provider: ModelType) -> BaseModel:
    if isinstance(model_or_provider, BaseModel):
        return model_or_provider
    return model_or_provider()


class DataFetcherProxy:
    def __init__(self, data_fetcher: _DataFetcher) -> None:
        self.data_fetcher = data_fetcher
        self.prefetched_batches = []

    def __iter__(self):
        return self.data_fetcher.__iter__()
    
    def __next__(self):
        if self.prefetched_batches:
            return self.prefetched_batches.pop(0)
        return next(self.data_fetcher)    
    
    def prefetch(self, n: int):
        while len(self.prefetched_batches) < n:
            x = next(self.data_fetcher.iterator)
            self.prefetched_batches.append(x)
        return self.prefetched_batches[:n]
    
    def reset(self):
        self.prefetched_batches.clear()
        return self.data_fetcher.reset(self)
    
    def __getattr__(self, name):
        return getattr(self.data_fetcher, name)
