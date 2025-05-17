from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

def sanitize_content(content) -> str:
    """Convert any content to a string appropriately."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    elif isinstance(content, (list, tuple)) and len(content) > 0:
        # For lists, we'll join strings or convert non-strings
        if all(isinstance(item, str) for item in content):
            return " ".join(content)
        else:
            # For mixed or numeric lists like [1985], convert first element
            return str(content[0]) if content else ""
    else:
        # For any other type, convert to string
        return str(content)

class AuthorInfo(BaseModel):
    collective: Optional[str] = None
    # lastname: Optional[str] = None
    firstname: Optional[str] = None
    # initials: Optional[str] = None

class SearchResult(BaseModel):
    """Represents a single search result."""
    title: str
    url: str
    content: str
    score: Optional[float] = None
    query: Optional[str] = None  # Track which query generated this result
    publication_date: Optional[str] = None
    authors: Optional[List[AuthorInfo]] = []

    @model_validator(mode='before')
    @classmethod
    def validate_content(cls, data):
        """Ensure content is always a string."""
        if isinstance(data, dict) and 'content' in data:
            data['content'] = sanitize_content(data['content'])
        return data


class SearchState(BaseModel):
    """Represents the state of the search process."""
    original_query: str
    generated_queries: List[str] = Field(default_factory=list)  # List of generated queries
    current_iteration: int = 0  # Track the iteration count for recursive search
    tavily_results: List[SearchResult] = Field(default_factory=list)
    pubmed_results: List[SearchResult] = Field(default_factory=list)  # Results from PubMed
    faiss_results: List[SearchResult] = Field(default_factory=list)
    bm25_results: List[SearchResult] = Field(default_factory=list)
    combined_results: List[SearchResult] = Field(default_factory=list)
    final_answer: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)  # Key points for the answer
    detailed_notes: Optional[str] = None  # Detailed notes for the answer
    confidence_score: Optional[float] = None
    knowledge_gaps: List[str] = Field(default_factory=list)  # Track knowledge gaps for further search
    historical_knowledge_gaps: List[str] = Field(default_factory=list)  # Track all previously identified knowledge gaps
    search_complete: bool = False  # Flag to indicate if search is complete
    metadata: Dict[str, Any] = Field(default_factory=dict)
    