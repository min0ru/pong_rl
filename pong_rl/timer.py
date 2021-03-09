import time


class ContextTimer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        print(f"[{self.name}] seconds: {self.total_time}")
