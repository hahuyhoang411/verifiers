import time
import random
import logging
from duckduckgo_search import DDGS
from brave import Brave
from typing import Optional, List, Dict, Any

from langchain_community.document_loaders import HuggingFaceDatasetLoader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
LANGCHAIN_AVAILABLE = True

# Initialize global variables for RAG components (lazy loading)
_rag_retriever: Optional[BM25Retriever] = None
_rag_data: Optional[List[Document]] = None
_rag_initialized: bool = False
_rag_init_error: Optional[str] = None

logger = logging.getLogger(__name__)

def _initialize_rag():
    """Initializes the RAG retriever components."""
    global _rag_retriever, _rag_data, _rag_initialized, _rag_init_error

    if _rag_initialized:
        return

    _rag_initialized = True # Mark as attempted initialization

    if not LANGCHAIN_AVAILABLE:
        _rag_init_error = "LangChain components not installed. Please run 'uv sync' after updating pyproject.toml."
        logger.error(_rag_init_error)
        return

    try:
        logger.info("Initializing RAG retriever for search_rag...")
        loader = HuggingFaceDatasetLoader(
            path="HoangHa/Tool-RL",
            name="corpus",
            page_content_column="context", # Use context for content
        )
        _rag_data = loader.load()
        logger.info(f"Loaded {len(_rag_data)} documents for RAG.")

        # Map context to page_content and add title/id to metadata
        # BM25Retriever uses page_content by default
        for doc in _rag_data:
             doc.metadata['title'] = doc.metadata.get('title', 'N/A')
             doc.metadata['id'] = doc.metadata.get('id', 'N/A')
             # page_content is already set by the loader

        _rag_retriever = BM25Retriever.from_documents(
            _rag_data,
            k=3 # Default k value
        )
        logger.info("BM25Retriever initialized successfully.")

    except Exception as e:
        _rag_init_error = f"Error during RAG initialization: {str(e)}"
        logger.error(_rag_init_error, exc_info=True)
        _rag_retriever = None
        _rag_data = None

def search_rag(query: str, num_results: int = 3) -> str:
    """Retrieves relevant documents from a pre-loaded corpus using BM25.

    Args:
        query: The search query string.
        num_results: Number of results to return (default: 3).

    Returns:
        Formatted string with bullet points of top results, each with title, id, and context from the corpus.

    Examples:
        {"name": "search_rag", "args": {"query": "effects of climate change on polar bears", "num_results": 2}}
    """
    # Ensure RAG components are initialized
    if not _rag_initialized:
        _initialize_rag()

    # Check for initialization errors
    if _rag_init_error:
        return f"Error: RAG system not initialized. {_rag_init_error}"
    if _rag_retriever is None:
        return "Error: RAG retriever is not available."

    try:
        # Set the number of results for this specific query
        _rag_retriever.k = min(num_results, 10) # Cap results

        results: List[Document] = _rag_retriever.invoke(query)

        if not results:
            return "No relevant documents found in the corpus."

        summaries = []
        for doc in results:
            title = doc.metadata.get('title', 'N/A')
            url = doc.metadata.get('id', 'N/A')
            content = doc.page_content

            #  # Truncate content reasonably
            # if len(content) > 300:
            #     period_index = content[:300].rfind('.')
            #     if period_index != -1:
            #         snippet = content[:period_index + 1]
            #     else:
            #         snippet = content[:300] + "..."
            # else:
            #     snippet = content

            summaries.append(content)

        return "\n\n".join(summaries)

    except Exception as e:
        logger.error(f"Error during RAG retrieval: {str(e)}", exc_info=True)
        return f"Error: {str(e)}"

def search_ddg(query: str, num_results: int = 5) -> str:
    """Searches DuckDuckGo and returns concise summaries of top results. Handles rate limiting with retries.

    Args:
        query: The search query string
        num_results: Number of results to return (default: 5, max: 10)

    Returns:
        Formatted string with bullet points of top results, each with title and brief summary, or an error message.

    Examples:
        {"query": "who invented the lightbulb", "num_results": 3}
    """
    max_retries = 10 # Increased retries
    initial_delay = 1.5 # Increased initial delay (seconds)
    backoff_factor = 2.0

    for attempt in range(max_retries + 1):
        try:
            # Use a context manager for DDGS to ensure cleanup
            with DDGS(timeout=10) as ddgs: # Added timeout to DDGS constructor
                # The actual search call that might raise an exception
                results = list(ddgs.text(query, max_results=min(num_results, 10)))

                # Process results if successful
                if not results:
                    return "No results found"

                summaries = []
                for r in results:
                    title = r.get('title', 'No Title') # Use .get for safety
                    body = r.get('body', '')
                    # Improve snippet generation robustness
                    if body:
                        # Try splitting by sentence, take first few, limit length
                        sentences = body.split('.')
                        snippet = '.'.join(sentences[:2]).strip() + '.' if len(sentences) > 1 else body
                        if len(snippet) > 250: # Slightly increased length
                             # Find last space before limit
                             last_space = snippet[:250].rfind(' ')
                             if last_space != -1:
                                 snippet = snippet[:last_space] + '...'
                             else: # No space found, just cut
                                 snippet = snippet[:250] + '...'
                        elif not snippet.endswith('.'):
                             snippet += '.'
                    else:
                         snippet = "No summary available."

                    summaries.append(f"• {title}\n  {snippet}")

                return "\n\n".join(summaries) # Success, return results

        except Exception as e:
            error_str = str(e)
            # More specific check for DDG rate limit errors
            # Common indicators: "Ratelimit", status codes like 429, or the specific URL mentioned
            is_rate_limit = (
                "Ratelimit" in error_str or
                "429" in error_str or # Common HTTP code for Too Many Requests
                "202" in error_str or # The code you observed
                "lite.duckduckgo.com" in error_str # Check if the error mentions the URL
            )

            if is_rate_limit:
                if attempt == max_retries:
                    logger.error(f"Search Error: Max retries ({max_retries}) exceeded for query '{query}'. Last error: {error_str}")
                    return f"Error: DuckDuckGo rate limit hit and max retries exceeded. ({error_str})"
                else:
                    # Calculate delay with exponential backoff and jitter
                    delay = initial_delay * (backoff_factor ** attempt)
                    delay += random.uniform(0, delay * 0.2) # Add up to 20% jitter
                    logger.warning(f"Search Info: Rate limit hit for query '{query}'. Retrying in {delay:.2f} seconds (Attempt {attempt + 1}/{max_retries}). Error: {error_str}")
                    time.sleep(delay)
                    # Continue to the next iteration of the loop for retry
            else:
                # It's not a rate limit error, return the error immediately
                logger.error(f"Search Error: Unexpected error for query '{query}'. Error: {error_str}")
                return f"Error: {error_str}"

    # Fallback if loop finishes without returning (shouldn't happen with current logic, but for safety)
    logger.error(f"Search Error: Failed to get search results for query '{query}' after all retries.")
    return "Error: Failed to get search results after multiple retries."
    
def search(query: str) -> str:
    """Searches the web and returns summaries of top results.
    
    Args:
        query: The search query string

    Returns:
        Formatted string with bullet points of top 3 results, each with title, source, url, and brief summary

    Examples:
        {"query": "who invented the lightbulb"} -> ["Thomas Edison (1847-1931) - Inventor of the lightbulb", ...]
        {"query": "what is the capital of France"} -> ["Paris is the capital of France", ...]
        {"query": "when was the Declaration of Independence signed"} -> ["The Declaration of Independence was signed on July 4, 1776", ...]
    """
    from brave import Brave
    from typing import List, Dict, Any

    try:
        brave = Brave()
        results = brave.search(q=query, count=10, raw=True) # type: ignore
        web_results = results.get('web', {}).get('results', []) # type: ignore
        
        if not web_results:
            return "No results found"

        summaries = []
        for r in web_results:
            if 'profile' not in r:
                continue
            header = f"{r['profile']['name']} ({r['profile']['long_name']})"
            title = r['title']
            snippet = r['description'][:300] + " ..."
            url = r['url'] 
            summaries.append(f"•  {header}\n   {title}\n   {snippet}\n   {url}")

        return "\n\n".join(summaries[:3])
    except Exception as e:
        return f"Error: {str(e)}"