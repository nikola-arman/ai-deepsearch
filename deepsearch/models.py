from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


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

import uuid

class SearchResult(BaseModel):
    """Represents a single search result."""
    id: int = Field(default_factory=lambda: uuid.uuid4().int & 0xFFFFFF)
    title: str
    url: str
    content: str
    score: Optional[float] = None
    query: Optional[str] = None  # Track which query generated this result    
    extracted_information: Optional[List[str]] = None
    is_url_credible: Optional[bool] = None
    
    @model_validator(mode='before')
    @classmethod
    def validate_content(cls, data):
        """Ensure content is always a string."""
        if isinstance(data, dict):
            if 'content' in data:
                data['content'] = sanitize_content(data['content'])

        return data
    
    @model_validator(mode='after')
    def validate_id(self):
        self.id = hash(self.url) & 0xFFFFFF
        return self

class SearchState(BaseModel):
    """State for the search process."""
    original_query: str
    generated_queries: List[str] = []
    current_iteration: int = 0
    search_complete: bool = False
    knowledge_gaps: List[str] = []
    historical_knowledge_gaps: List[str] = []
    tavily_results: List[SearchResult] = []
    brave_results: List[SearchResult] = []
    exa_results: list[SearchResult] = []
    exa_twitter_results: list[SearchResult] = []
    # Combined results from all search engines
    search_results: List[SearchResult] = []
    faiss_results: List[SearchResult] = []
    bm25_results: List[SearchResult] = []
    combined_results: List[SearchResult] = []
    verified_information: Dict[str, List[Dict[str, Any]]] = Field(default_factory=lambda: {
        "verified": [],
        "contradicted": [],
        "unverified": []
    })
    key_points: List[str] = []
    detailed_notes: Optional[str] = None
    final_answer: str = ""
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
