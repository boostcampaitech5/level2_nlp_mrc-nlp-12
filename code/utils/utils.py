import logging
import time
from contextlib import contextmanager
from pprint import pformat

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


def neat_logger(message: str) -> None:
    separator = '-' * 105
    log_message = pformat(message)
    logger.info(f"\n{separator}\n{log_message}\n{separator}")
