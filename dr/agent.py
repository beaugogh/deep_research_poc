import sys
from pathlib import Path
from state import State


class Agent:

    def __init__(self, state: State):
        self.state = state
        # get the directory of the current file
        module = sys.modules[self.__class__.__module__]
        self.cur_dir = Path(module.__file__).resolve().parent

    def run(self):
        raise NotImplementedError("Subclasses should implement this method")
