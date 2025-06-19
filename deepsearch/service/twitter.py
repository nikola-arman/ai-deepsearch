import string
import httpx
from typing import Union, Any, Dict, List, Set
import os
from ..schemas import (
    twitter,
    commons
)
from pydantic import ValidationError
import logging
import re
import random
from deepsearch.cache.wrapper import sqlite_cache, get_cached_value, set_cache_value

logger = logging.getLogger(__name__)

TWITTER_API_URL = os.getenv('TWITTER_API_URL', 'no-need').rstrip('/')
TWITTER_API_KEY = os.getenv('TWITTER_API_KEY', 'no-need')
TWITTER_USERNAME_TO_ID = "twitter_username_to_id"
# key builders
TIMEOUT_CFG = httpx.Timeout(60.0, connect=10.0) 


def tweet_key_builder(tweet_id: Union[str, int], **kwargs) -> str:
    return f"tweet:{tweet_id}"


def twitter_profile_key_builder(**kwargs) -> str:
    if 'user_id' in kwargs and kwargs['user_id']:
        user_id = kwargs['user_id']
        return f"twitter-profile:{user_id}"

    elif 'username' in kwargs and kwargs['username']:
        username = kwargs['username']
        user_id = get_cached_value(TWITTER_USERNAME_TO_ID, username)
        return f"twitter-profile:{user_id or username}"

    else:
        raise ValueError("Either user_id or username must be provided")


def twitter_tweet_key_builder_w_page(**kwargs) -> str:
    pagination_token = kwargs.get('pagination_token', '') 

    if 'user_id' in kwargs and kwargs['user_id']:
        user_id = kwargs['user_id']
        return f"twitter-profile:{user_id}-{pagination_token}"

    elif 'username' in kwargs and kwargs['username']:
        username = kwargs['username']
        user_id = get_cached_value(TWITTER_USERNAME_TO_ID, username)
        return f"twitter-profile:{user_id or username}-{pagination_token}"


@sqlite_cache(
    table_name="twitter",
    ttl_seconds=3600 * 6,
    key_prefix="tweet_info",
    key_builder=tweet_key_builder,
    object_builder=lambda obj_dict: commons.ResponseMessage[twitter.Tweet].model_validate(obj_dict)
)
def get_tweet_info(
    tweet_id: Union[str, int],
    twitter_api_base_url: str = TWITTER_API_URL,
    twitter_api_key: str = TWITTER_API_KEY
) -> commons.ResponseMessage[twitter.Tweet]:
    response_model = commons.ResponseMessage[twitter.Tweet]

    with httpx.Client(
        headers={
            "api-key": twitter_api_key,
        },
        timeout=TIMEOUT_CFG,
    ) as client: 
        try:
            res = client.get(
                f"{twitter_api_base_url}/tweets", 
                params={"ids": tweet_id},
            )
        except Exception as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )

        if res.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {res.status_code}: {res.text}"
            )

        data = res.json()
        results: dict[str, Any] = data['result']

        if str(tweet_id) not in results:
            return response_model(
                status=commons.APIStatus.NOT_FOUND, 
                error=f"Tweet {tweet_id} not found"
            )

        tweet_data = results[str(tweet_id)]['Tweet']

        try:
            tweet = twitter.Tweet.model_validate(tweet_data)
            return response_model(result=tweet)
        except ValidationError as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )


def get_top_followers(
    user_id: Union[str, int],
    twitter_api_base_url: str = TWITTER_API_URL,
    twitter_api_key: str = TWITTER_API_KEY
) -> commons.ResponseMessage[List[twitter.ConnectionCard]]:
    response_model = commons.ResponseMessage[List[twitter.ConnectionCard]]
    url = f"{twitter_api_base_url}/user/follower"
    
    with httpx.Client(
        headers={
            "api-key": twitter_api_key,
        },
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            res = client.get(url, params={"id": user_id})
        except Exception as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )

        if res.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {res.status_code}: {res.text}"
            )

        res_json: list[dict[str, Any]] = res.json().get('result', []) or []
        
        try:
            obj = [
                twitter.ConnectionCard.model_validate(x) 
                for x in res_json
            ]

            return response_model(result=obj)

        except ValidationError as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )
            

def get_top_following(
    user_id: Union[str, int],
    twitter_api_base_url: str = TWITTER_API_URL,
    twitter_api_key: str = TWITTER_API_KEY
) -> commons.ResponseMessage[List[twitter.ConnectionCard]]:
    response_model = commons.ResponseMessage[List[twitter.ConnectionCard]]
    url = f"{twitter_api_base_url}/user/{user_id}/following_v1"

    with httpx.Client(
        headers={
            "api-key": twitter_api_key,
        },
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            res = client.get(url)
        except Exception as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )
        
        if res.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {res.status_code}: {res.text}"
            )

        res_json: list[dict[str, Any]] = res.json().get('result', []) or []

        try:
            obj = [
                twitter.ConnectionCard.model_validate(x) 
                for x in res_json
            ]

            return response_model(result=obj)

        except ValidationError as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )


@sqlite_cache(
    table_name="twitter",
    ttl_seconds=3600 * 2,
    key_prefix="user_info",
    key_builder=twitter_profile_key_builder,
    object_builder=lambda obj_dict: commons.ResponseMessage[twitter.TwitterUserInfo].model_validate(obj_dict)
)
def get_twitter_user_info_by_id(
    user_id: Union[str, int],
    get_followers: bool = True,
    get_following: bool = True,
    twitter_api_base_url: str = TWITTER_API_URL, 
    twitter_api_key: str = TWITTER_API_KEY
) -> commons.ResponseMessage[twitter.TwitterUserInfo]:
    response_model = commons.ResponseMessage[twitter.TwitterUserInfo]
    url = f"{twitter_api_base_url}/user/{user_id}"

    with httpx.Client(
        headers={
            "api-key": twitter_api_key,
        },
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            res = client.get(url)
        except Exception as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )
        
        if res.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {res.status_code}: {res.text}"
            )
            
        res_json: dict[str, Any] = res.json()
        
        if not res_json.get('result'):
            return response_model(
                status=commons.APIStatus.NOT_FOUND, 
                error=f"User {user_id!r} not found"
            )
        
        try:
            obj = twitter.TwitterUserInfo.model_validate(res_json['result'])
            set_cache_value(TWITTER_USERNAME_TO_ID, obj.username, obj.id)

            if get_followers:
                followers_resp = get_top_followers(obj.id)
                obj.followers = followers_resp.result or []
            
            if get_following:
                following_resp = get_top_following(obj.id)
                obj.following = following_resp.result or []

            return response_model(result=obj)
        except ValidationError as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )


@sqlite_cache(
    table_name="twitter",
    ttl_seconds=3600 * 2,
    key_prefix="user_info",
    key_builder=twitter_profile_key_builder,
    object_builder=lambda obj_dict: commons.ResponseMessage[twitter.TwitterUserInfo].model_validate(obj_dict)
)
def get_twitter_user_info_by_username(
    username: str, 
    get_followers: bool = True,
    get_following: bool = True,
    twitter_api_base_url: str = TWITTER_API_URL, 
    twitter_api_key: str = TWITTER_API_KEY
) -> commons.ResponseMessage[twitter.TwitterUserInfo]:
    response_model = commons.ResponseMessage[twitter.TwitterUserInfo]
    url = f"{twitter_api_base_url}/user/by/username/{username}"
    
    with httpx.Client(
        headers={
            "api-key": twitter_api_key,
        },
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            res = client.get(url)
        except Exception as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )

        if res.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {res.status_code}: {res.text}"
            )

        res_json: dict[str, Any] = res.json()

        if not res_json.get('result'):
            return response_model(
                status=commons.APIStatus.NOT_FOUND, 
                error=f"User {username!r} not found"
            )

        user_info = res_json['result']
        
        try:
            user = twitter.TwitterUserInfo.model_validate(user_info)

            if get_followers:
                followers_resp = get_top_followers(user.id)
                user.followers = followers_resp.result or []
            
            if get_following:
                following_resp = get_top_following(user.id)
                user.following = following_resp.result or []

            return response_model(result=user)
        except ValidationError as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )


@sqlite_cache(
    table_name="twitter",
    ttl_seconds=3600 * 2,
    key_prefix="tweets",
    key_builder=twitter_tweet_key_builder_w_page,
    object_builder=lambda obj_dict: commons.ResponseMessage[twitter.TweetPage].model_validate(obj_dict)
)
def list_tweets_of_user(
    user_id: Union[str, int],
    pagination_token: str = "",
    twitter_api_base_url: str = TWITTER_API_URL,
    twitter_api_key: str = TWITTER_API_KEY
) -> commons.ResponseMessage[twitter.TweetPage]:
    response_model = commons.ResponseMessage[twitter.TweetPage]
    url = f"{twitter_api_base_url}/tweets/{user_id}"
    
    with httpx.Client(
        headers={
            "api-key": twitter_api_key,
        },
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            res = client.get(
                url, 
                params={
                   "pagination_token": pagination_token
                }
            )
        except Exception as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )
        
        if res.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {res.status_code}: {res.text}"
            )

        res_json: dict[str, Any] = res.json().get('result')

        try:
            obj = twitter.TweetPage.model_validate(res_json)
            return response_model(result=obj)

        except ValidationError as e:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )


is_valid_user = lambda user: (
    isinstance(user, twitter.TwitterUserInfo) 
    and user.id 
    and user.username
)

from queue import Queue


def build_twitter_social_graph(
    user_id: str, 
    max_depth: int = 1,
    max_expansion: int = 10,
    twitter_api_base_url: str = TWITTER_API_URL, 
    twitter_api_key: str = TWITTER_API_KEY,
) -> Dict[str, List[str]]:
    user_req = get_twitter_user_info_by_id(user_id, twitter_api_base_url, twitter_api_key)

    if not is_valid_user(user_req.result):
        return {}

    user = user_req.result

    graph: Dict[str, Set[str]] = {}

    users_map = {
        user.id: user
    }

    que = Queue() 
    que.put((user, 0))

    while not que.empty():
        user, depth = que.get()
        user: twitter.TwitterUserInfo

        if depth > max_depth:
            continue

        for follower in user.followers[:max_expansion]:
            if follower.rest_id not in users_map:
                info_req = get_twitter_user_info_by_id(follower.rest_id, twitter_api_base_url, twitter_api_key)

                if is_valid_user(info_req.result):
                    users_map[follower.rest_id] = info_req.result
                    que.put((info_req.result, depth + 1))

        for following in user.following[:max_expansion]:
            if following.rest_id not in users_map:
                info_req = get_twitter_user_info_by_id(following.rest_id, twitter_api_base_url, twitter_api_key)

                if is_valid_user(info_req.result):
                    users_map[following.rest_id] = info_req.result
                    que.put((info_req.result, depth + 1))

    logger.info(f"Built social graph for {user_id} with {len(users_map)} users")

    for id in users_map:
        graph[id] = set([])

    for id, user in users_map.items():
        for follower in user.followers: 
            if follower.rest_id in graph:
                graph[follower.rest_id].add(id)

            else:
                graph[follower.rest_id] = set([id])

        for following in user.following:
            if following.rest_id in graph:
                graph[following.rest_id].add(id)
            else:
                graph[following.rest_id] = set([id])

    return {
        k: list(v)
        for k, v in graph.items()
    }

from ..utils.misc import dsu


def get_tweet_threads_by_id(
    user_id: str,
    max_calls: int = 5,
    twitter_api_base_url: str = TWITTER_API_URL,
    twitter_api_key: str = TWITTER_API_KEY,
) -> commons.ResponseMessage[dict[str, list[twitter.Tweet]]]:

    response_model = commons.ResponseMessage[dict[str, list[twitter.Tweet]]]
    tweets: list[twitter.Tweet] = []

    current_page = ""

    for i in range(max_calls):
        req = list_tweets_of_user(
            user_id, 
            pagination_token=current_page,
            twitter_api_base_url=twitter_api_base_url,
            twitter_api_key=twitter_api_key
        ) 

        if req.result is None:
            logger.error(f"Error getting tweets for {user_id}: {req.error}")
            break

        tweets.extend(req.result.data)
        next_page = req.result.meta.next_token

        if current_page == next_page or next_page == "":
            break

        current_page = next_page

    map_idx = {
        val.id: i
        for i, val in enumerate(tweets)
    }

    relations = []

    for i, tweet in enumerate(tweets):
        for ref in (tweet.referenced_tweets or []):
            _type, _id = ref.get('type'), ref.get('id')

            if _type == "replied_to" and _id in map_idx:
                relations.append((map_idx[_id], i))

    parent = dsu(len(tweets), relations)
    unique_threads = set(parent)

    threads: dict[str, list[twitter.Tweet]] = {}

    for thread in unique_threads:
        threads[thread] = [
            tweets[i]
            for i in range(len(tweets))
            if parent[i] == thread
        ]

    for thread in threads:
        threads[thread].sort(key=lambda x: x.created_timestamp)

    return response_model(result=threads)


@sqlite_cache(
    table_name="twitter",
    ttl_seconds=3600 * 2,
    key_prefix="search_twitter_news",
    object_builder=lambda obj_dict: commons.ResponseMessage[twitter.TwitterSearchResult].model_validate(obj_dict)
)
def search_twitter_news(
    query: str,
    impression_count_limit=100,
    limit_api_results=50,
    use_raw=False,
    no_duplication=True,
) -> commons.ResponseMessage[twitter.TwitterSearchResult]:
    response_model = commons.ResponseMessage[twitter.TwitterSearchResult]
    if not use_raw:
        query = _optimize_twitter_query(
            query, remove_punctuations=True, token_limit=5, length_limit=30
        )
        logger.info(f"[search_twitter_news] Optimized query: {query}")

    if query.strip() == "":
        logger.error("[search_twitter_news] Empty query")
        return response_model(error="Empty query")

    url = f"{TWITTER_API_URL}/tweets/search/recent"

    params = {
        "query": f"{query} -is:retweet -is:reply -is:quote is:verified",
        "max_results": limit_api_results,
    }

    with httpx.Client(
        headers={"api-key": TWITTER_API_KEY},
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            resp = client.get(url, params=params)
        except Exception as e:
            logger.error(
                f"[search_twitter_news] Error occurred when calling api: {e}"
            )
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )

        if resp.status_code != 200:
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error {resp.status_code}: {resp.text}"
            )
        
        resp_json = resp.json()
        
    if resp_json.get("error") is not None:
        logger.error(
            "[search_twitter_news] Error occurred when calling api: "
            + resp_json["error"]["message"]
        )
        return response_model(
            status=commons.APIStatus.ERROR, 
            error="Error occurred when calling api",
        )

    search_result = twitter.TwitterSearchResult.model_validate(resp_json["result"])
    filtered_search_result = twitter.TwitterSearchResult(
        LookUps={},
        Meta=search_result.Meta,
    )

    hashes = set([])

    for id, item in search_result.LookUps.items():
        tweet = item.Tweet
        user = item.User

        if user is None:
            continue

        if (
            tweet.public_metrics.impression_count
            < impression_count_limit
        ):
            continue

        content_hash = hash(tweet.text)

        if no_duplication and content_hash in hashes:
            continue

        hashes.add(content_hash)

        filtered_search_result.LookUps[id] = item

    return response_model(result=filtered_search_result)


@sqlite_cache(
    table_name="twitter",
    ttl_seconds=3600 * 2,
    key_prefix="mentioned_tweets",
    object_builder=lambda obj_dict: commons.ResponseMessage[twitter.TweetPage].model_validate(obj_dict)
)
def get_mentioned_tweets(
    twitter_username: str,
    limit_results=50,
    min_author_followers=1000,
    replied=0,
    get_all=False,
) -> commons.ResponseMessage[twitter.TweetPage]:
    response_model = commons.ResponseMessage[twitter.TweetPage]
    
    if get_all:
        url = f"{TWITTER_API_URL}/user/by/username/{twitter_username}/mentions/all"
        params = {"max_results": 100}
    else:
        url = f"{TWITTER_API_URL}/user/by/username/{twitter_username}/mentions"
        params = {"replied": replied}

    with httpx.Client(
        headers={"api-key": TWITTER_API_KEY},
        timeout=TIMEOUT_CFG,
    ) as client:
        try:
            resp = client.get(url, params=params)
        except Exception as e:
            logger.error(
                f"[get_mentioned_tweets] Error occurred when calling api: {e}"
            )
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )

        if resp.status_code != 200:
            logger.error(
                f"[get_mentioned_tweets] Error occurred when calling api: {resp.status_code}, url: {resp.url}"
            )
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error occurred when calling api: {resp.status_code}, url: {resp.url}"
            )

        resp_json = resp.json()
        if resp_json.get("error"):
            logger.error(
                f"[get_mentioned_tweets] Error occurred when calling API: {resp_json['error']['message']}, url: {resp.url}"
            )
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=f"Error occurred when calling API: {resp_json['error']['message']}, url: {resp.url}"
            )

        try:
            tweet_page = twitter.TweetPage.model_validate(resp_json["result"])
        except ValidationError as e:
            logger.error(
                f"[get_mentioned_tweets] Error occurred when validating response: {e}"
            )
            return response_model(
                status=commons.APIStatus.ERROR, 
                error=str(e)
            )
        
        filtered_tweets = []
        for tweet in tweet_page.data:
            user_info_response = get_twitter_user_info_by_id(user_id=tweet.author_id, get_followers=False, get_following=False)
            if user_info_response.status != commons.APIStatus.OK:
                continue
            
            author_info = user_info_response.result
            if author_info.public_metrics.followers_count >= min_author_followers:
                filtered_tweets.append(tweet)

        filtered_tweet_page = twitter.TweetPage(
            data=filtered_tweets[:limit_results],
            meta=tweet_page.meta
        )

        return response_model(result=filtered_tweet_page)


def _optimize_twitter_query(
    query: str,
    remove_punctuations=False,
    token_limit=-1,
    pat: re.Pattern = None,
    length_limit=30,
) -> str:
    and_token = re.compile(r"\bAND\b", flags=re.IGNORECASE)
    spacing = re.compile(r"\s+")

    query = and_token.sub(" ", query)
    query = spacing.sub(" ", query)

    tokenized_query = re.split(r"\bor\b", query, flags=re.IGNORECASE)
    filtered_tokenized_query = []

    if pat is not None:
        tokenized_query = [
            i.strip() for i in tokenized_query if pat.fullmatch(i.strip())
        ]

    # sort and remove duplicates
    tokenized_query = sorted(tokenized_query, key=len, reverse=True)

    for i in tokenized_query:
        i = i.strip(" '\"")

        if remove_punctuations:
            i = "".join([c for c in i if c not in string.punctuation])

        if len(filtered_tokenized_query) == 0:
            filtered_tokenized_query.append(i)
        else:
            if any([i.lower() in x.lower() for x in filtered_tokenized_query]):
                continue
            else:
                filtered_tokenized_query.append(i)

    random.shuffle(filtered_tokenized_query)

    if token_limit != -1:
        filtered_tokenized_query = filtered_tokenized_query[:token_limit]

    if len(filtered_tokenized_query) == 0:
        return ""

    res = ""
    for item in filtered_tokenized_query:
        if len(res) + len(item) > length_limit:
            break

        if len(res) > 0:
            res += " OR "

        res += item

    if len(res) == 0:
        e = tokenized_query[0].split()

        for ee in e:
            if len(res) + len(ee) > length_limit:
                break

            if len(res) > 0:
                res += " "

            res += ee

    return res