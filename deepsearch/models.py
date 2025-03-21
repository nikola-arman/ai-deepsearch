from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """Represents a single search result."""
    title: str
    url: str
    content: str
    score: Optional[float] = None
    query: Optional[str] = None  # Track which query generated this result


class SearchState(BaseModel):
    """Represents the state of the search process."""
    original_query: str
    refined_query: Optional[str] = None
    generated_queries: List[str] = Field(default_factory=list)  # List of generated queries
    current_iteration: int = 0  # Track the iteration count for recursive search
    tavily_results: List[SearchResult] = Field(default_factory=list)
    faiss_results: List[SearchResult] = Field(default_factory=list)
    bm25_results: List[SearchResult] = Field(default_factory=list)
    combined_results: List[SearchResult] = Field(default_factory=list)
    final_answer: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)  # Key points for the answer
    detailed_notes: Optional[str] = None  # Detailed notes for the answer
    confidence_score: Optional[float] = None
    knowledge_gaps: List[str] = Field(default_factory=list)  # Track knowledge gaps for further search
    search_complete: bool = False  # Flag to indicate if search is complete
    metadata: Dict[str, Any] = Field(default_factory=dict)