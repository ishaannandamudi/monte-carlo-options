"""Small, readable logging setup for the API boundary."""

import logging


def configure_logging(level: str) -> None:
    """Configure process logging without attaching duplicate handlers."""

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
        force=True,
    )
