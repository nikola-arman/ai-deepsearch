from typing import Dict
from deepsearch.magic import retry
from deepsearch.schemas.agents import SearchResult
from deepsearch.schemas import twitter, commons, agents
from deepsearch.service.twitter import get_twitter_user_info_by_id, get_twitter_user_info_by_username, list_tweets_of_user, search_twitter_news, get_mentioned_tweets
import logging

from deepsearch.utils.misc import truncate_text
logger = logging.getLogger(__name__)


def get_twitter_data_by_username(username: str) -> agents.TwitterData:
    def _get_user_info() -> twitter.TwitterUserInfo:
        user_info_response = get_twitter_user_info_by_username(username=username, get_followers=False, get_following=False)
        if user_info_response.status == commons.APIStatus.OK:
            return user_info_response.result
        raise Exception(f"Error getting user info for {username}: {user_info_response.error}")
    
    def _get_recent_tweets() -> twitter.TweetPage:
        recent_tweets_response = list_tweets_of_user(user_id=user_info.id)
        if recent_tweets_response.status == commons.APIStatus.OK:
            return recent_tweets_response.result
        raise Exception(f"Error getting recent tweets for {username}: {recent_tweets_response.error}")
    
    def _get_mentioned_tweets() -> twitter.TweetPage:
        mentioned_tweets_response = get_mentioned_tweets(twitter_username=username, limit_results=10, min_author_followers=1000)
        if mentioned_tweets_response.status == commons.APIStatus.OK:
            return mentioned_tweets_response.result
        raise Exception(f"Error getting mentioned tweets for {username}: {mentioned_tweets_response.error}")
        
    twitter_data = {}
    try:
        user_info: twitter.TwitterUserInfo = retry(_get_user_info, max_retry=3, first_interval=2, interval_multiply=2)()
        twitter_data["user_info"] = user_info
    except Exception as e:
        logger.error(f"Error getting user info for {username}: {str(e)}", exc_info=True)
        raise Exception(f"Error getting user info for {username}: {str(e)}")
    
    try:
        recent_tweets: twitter.TweetPage = retry(_get_recent_tweets, max_retry=3, first_interval=2, interval_multiply=2)()
        twitter_data["recent_tweets"] = recent_tweets
    except Exception as e:
        logger.warning(f"Error getting recent tweets for {username}: {str(e)}", exc_info=True)
    
    logger.info(f"Getting mentioned tweets for {username}")
    logger.info(twitter_data.model_dump_json(indent=2))

    try:
        mentioned_tweets: twitter.TweetPage = retry(_get_mentioned_tweets, max_retry=3, first_interval=2, interval_multiply=2)()
        twitter_data["mentioned_tweets"] = mentioned_tweets
    except Exception as e:
        logger.warning(f"Error getting mentioned tweets for {username}: {str(e)}", exc_info=True)
    
    return agents.TwitterData.model_validate(twitter_data)


def twitter_context_to_search_result(twitter_context: Dict[str, agents.TwitterData]) -> list[SearchResult]:
    search_results = []
    for username, twitter_data in twitter_context.items():
        user_info = twitter_data.user_info

        twitter_profile = f"""
- Username: {user_info.username}
- Name: {user_info.name}
- Description: {user_info.description}
- Followers Count: {user_info.public_metrics.followers_count}
- Tweets Count: {user_info.public_metrics.tweet_count}
"""
        
        search_results.append(SearchResult(
            title=f"{user_info.name} (@{user_info.username}) / X",
            url=f"https://x.com/{user_info.username}",
            content=twitter_profile,
            score=1.0,
        ))

        if twitter_data.recent_tweets:
            for tweet in twitter_data.recent_tweets.data:
                search_results.append(SearchResult(
                    title=f"{user_info.name} on X: \"{truncate_text(tweet.text)}\"",
                    url=f"https://x.com/{user_info.username}/status/{tweet.id}",
                    content=tweet.text,
                    score=1.0,
                ))

        if twitter_data.mentioned_tweets:
            for tweet in twitter_data.mentioned_tweets.data:
                user_info_response = get_twitter_user_info_by_id(user_id=tweet.author_id, get_followers=False, get_following=False)
                if user_info_response.status != commons.APIStatus.OK:
                    continue
                
                author_info = user_info_response.result
                
                search_results.append(SearchResult(
                    title=f"{author_info.name} on X: \"{truncate_text(tweet.text)}\"",
                    url=f"https://x.com/{author_info.username}/status/{tweet.id}",
                    content=tweet.text,
                    score=1.0,
                ))

    return search_results


def twitter_search(query: str) -> list[SearchResult]:
    twitter_search_response = search_twitter_news(
        query=query,
        impression_count_limit=1000,
        limit_api_results=10,
    )
    if twitter_search_response.status != commons.APIStatus.OK:
        raise Exception(f"Error searching twitter: {twitter_search_response.error}")
    
    search_results = []
    for tweet_id, tweet_data in twitter_search_response.result.LookUps.items():
        tweet = tweet_data.Tweet
        user = tweet_data.User
        
        search_results.append(SearchResult(
            title=f"{user.name} on X: \"{truncate_text(tweet.text)}\"",
            url=f"https://x.com/{user.username}/status/{tweet.id}",
            content=tweet.text,
            query=query,
            score=1.0,
        ))

    return search_results
