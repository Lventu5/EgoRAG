import logging
import sys

class LevelAwareFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.WARNING:
            # Per WARNING ed ERROR -> mostra [LEVEL]
            fmt = "[%(levelname)s] %(message)s"
        else:
            # Per INFO e DEBUG -> solo messaggio
            fmt = "%(message)s"
        formatter = logging.Formatter(fmt)
        return formatter.format(record)
