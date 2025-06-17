from pydantic import BaseModel

class Generator:
    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        self.value = yield from self.gen
        return self.value


def escape_dollar_signs(text: str) -> str:
    """
    Escapes all dollar signs in the text by replacing them with backslash-dollar.
    This prevents markdown from interpreting them as math formula delimiters.

    Args:
        text: The text to process

    Returns:
        Text with all dollar signs escaped
    """
    return text.replace('$', '\\$')


def get_url_domain(url: str) -> str:
    """
    Get the prefix of a URL
    """
    if 'https://' in url:
        url = url.split('https://')[1]
    url = url.split('/')[0]
    return url

def dsu(n: int, relations: list[tuple[int, int]]) -> list[int]:
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def unite(x: int, y: int) -> None:
        x_root = find(x)
        y_root = find(y)

        if x_root == y_root:
            return

        if rank[x] < rank[y]:
            parent[x_root] = y_root
        
        elif rank[x] > rank[y]:
            parent[y_root] = x_root
        
        else:
            parent[y_root] = x_root
            rank[x_root] += 1

    for x, y in relations:
        unite(x, y)

    return parent

def truncate_text(text: str, max_length: int = 100) -> str:
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text