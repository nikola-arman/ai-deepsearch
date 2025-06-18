from functools import lru_cache
import redis
from redis import asyncio as aredis
from typing import Callable
import deepsearch.constants as const
import logging
import pickle
import asyncio

logger = logging.getLogger(__name__)

redis_cfg = dict(
    host=const.REDIS_HOST,
    port=const.REDIS_PORT,
    db=const.REDIS_DB,
    password=const.REDIS_PASSWORD
)


@lru_cache(maxsize=1)
def get_redis_connection_pool():
    global redis_cfg
    return redis.ConnectionPool(**redis_cfg)


@lru_cache(maxsize=1)
def get_aio_redis_connection_pool():
    global redis_cfg
    return aredis.ConnectionPool(**redis_cfg)


def cache_for(interval_seconds: float) -> Callable:
    def get_parameters_hash(*args, **kwargs):
        return hash((*args, *sorted(kwargs.items())))

    def decorator(func: Callable) -> Callable:
        fn_name = func.__name__
        fn_module = func.__module__
        redis_cache_key = f"{fn_module}.{fn_name}:result"

        def sync_wrapper(*args, **kwargs):
            nonlocal redis_cache_key

            key = f"{redis_cache_key}:{get_parameters_hash(*args, **kwargs)}"
            
            with redis.Redis(connection_pool=get_redis_connection_pool()) as redis_client:
                if redis_client is not None:
                    pickle_str = redis_client.get(key)

                    if pickle_str is not None:
                        logger.info(
                            f"Cache hit for {key}; call {fn_module}.{fn_name}(args={args}, kwargs={kwargs})"
                        )
                        return pickle.loads(pickle_str)

                res = func(*args, **kwargs)

                if redis_client is not None:
                    pickle_str = pickle.dumps(res)
                    redis_client.set(key, pickle_str, ex=interval_seconds)

                return res

        async def async_wrapper(*args, **kwargs):
            nonlocal redis_cache_key

            key = f"{redis_cache_key}:{get_parameters_hash(*args, **kwargs)}"
            
            async with aredis.Redis(
                connection_pool=get_aio_redis_connection_pool()
            ) as redis_client:
                if redis_client is not None:
                    pickle_str = await redis_client.get(key)

                    if pickle_str is not None:
                        logger.info(
                            f"Cache hit for {key}; call {fn_module}.{fn_name}(args={args}, kwargs={kwargs})"
                        )
                        return pickle.loads(pickle_str)

                res = await func(*args, **kwargs)

                if redis_client is not None:
                    pickle_str = pickle.dumps(res)
                    await redis_client.set(key, pickle_str, ex=interval_seconds)

            return res

        if asyncio.iscoroutinefunction(func):
            return async_wrapper

        return sync_wrapper

    return decorator
