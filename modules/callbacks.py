from queue import Queue
from threading import Thread

import torch
import transformers

import modules.shared as shared


# Copied from https://github.com/PygmalionAI/gradio-ui/
class StopWhenIdsAreFound(transformers.StoppingCriteria):
    def __init__(self, ids_to_watch: torch.LongTensor, starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.ids_to_watch = ids_to_watch
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]
            # Can't unfold, output is still too tiny. Skip.
            if trimmed_sample.shape[-1] < self.ids_to_watch.shape[-1]:
                continue

            for window in trimmed_sample.unfold(0, self.ids_to_watch.shape[-1], 1):
                if torch.all(torch.eq(self.ids_to_watch, window)):
                    return True
        return False

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids)
        return False

class Iteratorize:

    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, extra_arguments={}):
        self.actual_function = func
        self.data_queue = Queue(maxsize=1)
        self.end_signal = object()
        self.extra_arguments = extra_arguments

        def _data_callback(value):
            self.data_queue.put(value)

        def task_generator():
            ret = self.actual_function(callback=_data_callback, **self.extra_arguments)
            self.data_queue.put(self.end_signal)

        Thread(target=task_generator).start()

    def __iter__(self):
        return self

    def __next__(self):
        data_object = self.data_queue.get(True, None)
        if data_object is self.end_signal:
            raise StopIteration
        else:
            return data_object
