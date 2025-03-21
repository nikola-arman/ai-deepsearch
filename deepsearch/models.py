from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a single search result."""
    title: str
    url: str
    content: str
    score: Optional[float] = None


class SearchState(BaseModel):
    """Represents the state of the search process."""
    original_query: str
    refined_query: Optional[str] = None
    tavily_results: List[SearchResult] = Field(default_factory=list)
    faiss_results: List[SearchResult] = Field(default_factory=list)
    bm25_results: List[SearchResult] = Field(default_factory=list)
    combined_results: List[SearchResult] = Field(default_factory=list)
    final_answer: Optional[str] = None
    confidence_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)