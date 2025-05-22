import time
import traceback
from typing import Callable
import asyncio

import logging

logger = logging.getLogger(__name__)


def retry(func: Callable, max_retry=5, first_interval=2, interval_multiply=1):
    def sync_wrapper(*args, **kwargs):
        interval = first_interval
        for iter in range(max_retry + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as err:
                traceback.print_exc()
                logger.error(
                    f"Function {func.__name__} failed with error '{err}'. Retry attempt {iter}/{max_retry}"
                )

                if iter == max_retry:
                    raise err

            time.sleep(interval)
            interval *= interval_multiply

    async def async_wrapper(*args, **kwargs):
        interval = first_interval
        for iter in range(max_retry + 1):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as err:
                traceback.print_exc()
                logger.error(
                    f"Function {func.__name__} failed with error '{err}'. Retry attempt {iter}/{max_retry}"
                )
                
                if iter == max_retry:
                    raise err
                
            await asyncio.sleep(interval)
            interval *= interval_multiply

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
