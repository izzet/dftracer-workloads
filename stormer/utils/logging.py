import os
import logging
import time
import inspect
import sys
import platform

from strenum import StrEnum

logger = logging.getLogger(__name__)

# https://medium.com/@ryan_forrester_/adding-color-to-python-terminal-output-a-complete-guide-147fcb1c335f
def _enable_windows_color():
    """Enable color support on Windows"""
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        return True
    except Exception:
        return False


# https://medium.com/@ryan_forrester_/adding-color-to-python-terminal-output-a-complete-guide-147fcb1c335f
def _check_color_supports():
    """Check if the current terminal supports colors"""
    # Force colors if FORCE_COLOR env variable is set
    if os.getenv("FORCE_COLOR"):
        return True

    # Check platform-specific support
    if platform.system() == "Windows":
        return _enable_windows_color()

    # Most Unix-like systems support colors
    return os.getenv("TERM") not in ("dumb", "")


color_supported = _check_color_supports()


# https://stackoverflow.com/a/55612356
class FormatterNanosecond(logging.Formatter):
    default_nsec_format = "%s,%09d"

    def formatTime(self, record, datefmt=None):
        if datefmt is not None:
            return super().formatTime(record, datefmt)
        ct = self.converter(record.created_ns / 1e9)
        t = time.strftime(self.default_time_format, ct)
        s = self.default_nsec_format % (t, record.created_ns - (record.created_ns // 10**9) * 10**9)
        return s


class ColoredFormatter(logging.Formatter):
    _green = "\x1b[32;20m"
    _purple = "\x1b[35;20m"
    _yellow = "\x1b[33;20m"
    _red = "\x1b[31;20m"
    _bold_red = "\x1b[31;1m"
    _reset = "\x1b[0m"
    _blue = "\x1b[34;20m"
    _magenta = "\x1b[35;20m"
    _cyan = "\x1b[36;20m"
    _white = "\x1b[37;20m"
    _orange = "\x1b[38;5;208m"
    _prefix_format = "%(levelname).1s"
    _time_format = "%(asctime)s"

    LEVEL_FMT = {
        logging.DEBUG: _cyan + _prefix_format + _reset,
        logging.INFO: _green + _prefix_format + _reset,
        logging.WARNING: _yellow + _prefix_format + _reset,
        logging.ERROR: _red + _prefix_format + _reset,
        logging.CRITICAL: _bold_red + _prefix_format + _reset,
    }

    @staticmethod
    def process_format(fmt: str, levelno) -> str:
        fmt = fmt.replace("^lvl^", ColoredFormatter.LEVEL_FMT.get(levelno))
        fmt = fmt.replace("^time^", ColoredFormatter._time_format)
        if not color_supported:
            fmt = fmt.replace("^green^", "")
            fmt = fmt.replace("^purple^", "")
            fmt = fmt.replace("^yellow^", "")
            fmt = fmt.replace("^red^", "")
            fmt = fmt.replace("^bold_red^", "")
            fmt = fmt.replace("^reset^", "")
            fmt = fmt.replace("^blue^", "")
            fmt = fmt.replace("^magenta^", "")
            fmt = fmt.replace("^cyan^", "")
            fmt = fmt.replace("^white^", "")
            return fmt
        # to enable color, user should set following format
        # ^color^ [%(levelname).1s] ^reset^ %(asctime)s - %(message)s
        fmt = fmt.replace("^green^", ColoredFormatter._green)
        fmt = fmt.replace("^purple^", ColoredFormatter._purple)
        fmt = fmt.replace("^yellow^", ColoredFormatter._yellow)
        fmt = fmt.replace("^red^", ColoredFormatter._red)
        fmt = fmt.replace("^bold_red^", ColoredFormatter._bold_red)
        fmt = fmt.replace("^reset^", ColoredFormatter._reset)
        fmt = fmt.replace("^blue^", ColoredFormatter._blue)
        fmt = fmt.replace("^magenta^", ColoredFormatter._magenta)
        fmt = fmt.replace("^cyan^", ColoredFormatter._cyan)
        fmt = fmt.replace("^white^", ColoredFormatter._white)
        return fmt

    def format(self, record):
        # log_fmt = self.LEVEL_FMT.get(record.levelno)
        # log_fmt += self._green + self._time_format + self._reset + self._fmt
        log_fmt = self.process_format(fmt=self._fmt, levelno=record.levelno)
        formatter = FormatterNanosecond(log_fmt)
        return formatter.format(record)


# https://stackoverflow.com/a/55612356
class LogRecordNanosecond(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        self.created_ns = time.time_ns()  # Fetch precise timestamp
        super().__init__(*args, **kwargs)


logging.setLogRecordFactory(LogRecordNanosecond)

class LogLevel(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

def loglevel_to_logging(level: LogLevel) -> int:
    mapping = {
        LogLevel.DEBUG: logging.DEBUG,
        LogLevel.INFO: logging.INFO,
        LogLevel.WARNING: logging.WARNING,
        LogLevel.ERROR: logging.ERROR,
        LogLevel.CRITICAL: logging.CRITICAL,
    }
    return mapping.get(level, logging.INFO)

def configure_logging(log_level: LogLevel = LogLevel.INFO, output_dir: str | None = None):
    logfile_path = None
    if output_dir:
        logfile_path = os.path.join(output_dir, "runtime.log")

    log_level_config = log_level
    log_level = loglevel_to_logging(log_level_config)

    fmt = "[%(levelname).1s][%(asctime)s] %(message)s"
    handlers = []

    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(log_level)
    # stdout_handler.setFormatter(ColoredFormatter("%(message)s [%(pathname)s:%(lineno)d"))
    # stdout_handler.setFormatter(ColoredFormatter("[^lvl^][^blue^^time^^reset^] ^yellow^%(message)s^reset^ ^purple^[%(pathname)s:%(lineno)d]^reset^"))
    stdout_handler.setFormatter(
        ColoredFormatter("[^lvl^][^blue^^time^^reset^] ^yellow^%(message)s^reset^")
    )

    handlers.append(stdout_handler)

    if logfile_path:
        # fmt = "[%(levelname).1s][%(asctime)s] %(message)s [%(pathname)s:%(lineno)d]"
        file_handler = logging.FileHandler(logfile_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(FormatterNanosecond(fmt))
        handlers.append(file_handler)

    logging.basicConfig(
        level=log_level,
        force=True,
        handlers=handlers,
        format=fmt,
    )


__all__ = [
    "ColoredFormatter",
    "FormatterNanosecond",
    "configure_logging",
    "LogLevel",
]
