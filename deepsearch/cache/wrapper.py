import json
from typing import Any, Callable, Optional, TypeVar
import logging
import pickle
import asyncio

from pydantic import BaseModel
from .async_cache import get_async_sqlite_cache
from .sync_cache import get_sqlite_cache

T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])

logger = logging.getLogger(__name__)

def sqlite_cache(
    table_name: str,
    ttl_seconds: int = 3600 * 24 * 30, # 30 days
    key_prefix: str = "",
    key_builder: Optional[Callable[..., str]] = None,
    object_builder: Optional[Callable[..., Any]] = None        
) -> Callable:
    def get_parameters_hash(*args, **kwargs):
        return hash((*args, *sorted(kwargs.items())))

    def decorator(func: Callable) -> Callable:
        def sync_wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = f"{key_prefix}:{key_builder(*args, **kwargs)}"
            else:
                # Default key building from function name and arguments
                arg_str = ":".join(str(arg) for arg in args)
                kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"
                        
            cache_client = get_sqlite_cache()
            if cache_client is not None:
                data_str = cache_client.get(table_name, cache_key)

                if data_str is not None:
                    logger.info(
                        f"Cache hit for {cache_key}; call {func.__name__}(args={args}, kwargs={kwargs})"
                    )
                    obj_dict = json.loads(data_str)
                    if object_builder:
                        return object_builder(obj_dict)
                    else:
                        return obj_dict

            res = func(*args, **kwargs)

            if cache_client is not None:
                if isinstance(res, BaseModel):
                    data_str = res.model_dump_json()
                else:
                    data_str = json.dumps(res)
                cache_client.set(table_name, cache_key, data_str, ttl=ttl_seconds)

            return res

        async def async_wrapper(*args, **kwargs):
            # Build cache key
            if key_builder:
                cache_key = f"{key_prefix}:{key_builder(*args, **kwargs)}"
            else:
                # Default key building from function name and arguments
                arg_str = ":".join(str(arg) for arg in args)
                kwarg_str = ":".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = f"{key_prefix}:{func.__name__}:{arg_str}:{kwarg_str}"
            
            cache_client = get_async_sqlite_cache()
            if cache_client is not None:
                data_str = await cache_client.get(table_name, cache_key)

                if data_str is not None:
                    logger.info(
                        f"Cache hit for {cache_key}; call {func.__name__}(args={args}, kwargs={kwargs})"
                    )
                    obj_dict = json.loads(data_str)
                    if object_builder:
                        return object_builder(obj_dict)
                    else:
                        return obj_dict

            res = await func(*args, **kwargs)

            if cache_client is not None:
                if isinstance(res, BaseModel):
                    data_str = res.model_dump_json()
                else:
                    data_str = json.dumps(res)
                await cache_client.set(table_name, cache_key, data_str, ttl=ttl_seconds)

            return res

        if asyncio.iscoroutinefunction(func):
            return async_wrapper

        return sync_wrapper

    return decorator


def set_cache_value(
    table_name: str, 
    key: str, 
    value: Any, 
    ttl_seconds: int = 3600 * 24 * 30,
    force_update: bool = False,
) -> None:
    cache_client = get_sqlite_cache()
    if cache_client is None:
        return
    
    if not force_update and cache_client.get(table_name, key):
        return

    cache_client.set(table_name, key, value, ttl=ttl_seconds)


def get_cached_value(table_name: str, key: str) -> Any:
    cache_client = get_sqlite_cache()
    if cache_client is None:
        return None
    
    return cache_client.get(table_name, key)
