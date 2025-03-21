from typing import Any
from mcp.server.fastmcp import FastMCP
import os
import asyncio
import praw
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize FastMCP server
mcp = FastMCP("reddit")

# Create a global Reddit instance using credentials from the environment
reddit = praw.Reddit(
    client_id=os.environ.get("REDDIT_CLIENT_ID"),
    client_secret=os.environ.get("REDDIT_CLIENT_SECRET"),
    user_agent=os.environ.get("REDDIT_USER_AGENT")
)

def compute_threshold(time_filter: str) -> Any:
    """
    Compute a UNIX timestamp threshold based on a natural language time filter.
    
    Parameters:
    - time_filter: A string such as "hour", "day", "week", "month", or "year". If "all", returns None.
    
    Returns:
    - A UNIX timestamp corresponding to the start of the period, or None if time_filter is "all" or unrecognized.
    """
    now = time.time()
    if time_filter == "hour":
        return now - 3600
    elif time_filter == "day":
        return now - 86400
    elif time_filter == "week":
        return now - (7 * 86400)
    elif time_filter == "month":
        return now - (30 * 86400)
    elif time_filter == "year":
        return now - (365 * 86400)
    else:
        return None

# @mcp.tool()
async def get_post_details(submission):
    """
    Extract and return key details from a submission.
    
    Parameters:
    - submission: A PRAW submission object.
    
    Returns:
    - A dictionary containing post details such as id, title, selftext, URL, score, comment count, and creation time.
    """
    return {
        'id': submission.id,
        'title': submission.title,
        'selftext': submission.selftext,
        'url': submission.url,
        'score': submission.score,
        'num_comments': submission.num_comments,
        'created_utc': submission.created_utc,
    }

@mcp.tool()
async def search_posts(query: str, subreddit: str = 'all', sort: str = 'relevance',
                       time_filter: str = 'all', limit: int = 10):
    """
    Asynchronously search for posts in a specified subreddit using a natural language time filter.
    
    This function now calls get_post_details for each found submission so that the returned list
    contains dictionaries with post details.
    
    Parameters:
    - query: The search query (e.g., "Nvidia new product").
    - subreddit: The subreddit name to search in (default is 'all').
    - sort: The sort order (e.g., 'relevance', 'hot', 'top').
    - time_filter: A natural language filter such as "hour", "day", "week", "month", or "year". If "all", no additional filtering is done.
    - limit: Maximum number of posts to return.
    
    Returns:
    - A list of dictionaries, each containing the details of a submission created within the specified time window.
    """
    print("Searching:", query)
    subreddit_instance = reddit.subreddit(subreddit)
    # Run the blocking search call in a thread to preserve async behavior.
    submissions = await asyncio.to_thread(
        lambda: list(subreddit_instance.search(query, sort=sort, time_filter=time_filter, limit=limit))
    )
    
    # Compute threshold based on natural language filter.
    threshold = compute_threshold(time_filter)
    if threshold is not None:
        submissions = [submission for submission in submissions if submission.created_utc >= threshold]
    
    # Directly call get_post_details for each submission.
    details_list = []
    for submission in submissions:
        details = await get_post_details(submission)
        details_list.append(details)
    return details_list

@mcp.tool()
async def get_submission_comments(submission, limit: int = 20):
    """
    Retrieve a limited number of comments from a submission.
    
    Parameters:
    - submission: A PRAW submission object.
    - limit: Maximum number of comments to retrieve.
    
    Returns:
    - A list of dictionaries, each containing comment id, body, score, and creation time.
    """
    # Use asyncio.to_thread to run blocking calls in a separate thread.
    await asyncio.to_thread(submission.comments.replace_more, limit=0)
    comments_list = await asyncio.to_thread(lambda: submission.comments.list())
    results = []
    count = 0
    for comment in comments_list:
        if count >= limit:
            break
        results.append({
            'id': comment.id,
            'body': comment.body,
            'score': comment.score,
            'created_utc': comment.created_utc,
        })
        count += 1
    return results

@mcp.tool()
async def search_comments_in_posts(query: str, subreddit: str = 'all', post_limit: int = 10,
                                   comment_limit: int = 20, time_filter: str = 'all'):
    """
    Asynchronously search for posts matching the query and extract a set number of comments from each post.
    
    Since search_posts now returns post details (rather than submission objects), we re-fetch each submission
    by ID before retrieving its comments.
    
    Parameters:
    - query: The search query.
    - subreddit: The subreddit to search in (default is 'all').
    - post_limit: Maximum number of posts to search.
    - comment_limit: Maximum number of comments to retrieve per post.
    - time_filter: A natural language filter like "hour", "day", "week", "month", or "year". Defaults to "all".
    
    Returns:
    - A list of dictionaries where each dictionary contains post details and its associated comments.
    """
    posts_details = await search_posts(query, subreddit=subreddit, sort="relevance",
                                       time_filter=time_filter, limit=post_limit)
    results = []
    for details in posts_details:
        # Retrieve the full submission object by ID so that comments can be fetched.
        submission = await asyncio.to_thread(lambda: reddit.submission(id=details['id']))
        comments = await get_submission_comments(submission, limit=comment_limit)
        details['comments'] = comments
        results.append(details)
    return results

if __name__ == "__main__":
    mcp.run(transport='stdio')
