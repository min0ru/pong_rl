import time


class ContextTimer:
    def __init__(self, name, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, *args, **kwargs):
        self.end_time = time.time()
        self.total_time = self.end_time - self.start_time
        message = f"[{self.name}] seconds: {self.total_time}"
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
