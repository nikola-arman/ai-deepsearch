import os
import json
from urllib import parse
import base64
import sys

"""Filename matching with shell patterns.

fnmatch(FILENAME, PATTERN) matches according to the local convention.
fnmatchcase(FILENAME, PATTERN) always takes case in account.

The functions operate by translating the pattern into a regular
expression.  They cache the compiled regular expressions for speed.

The function translate(PATTERN) returns a regular expression
corresponding to PATTERN.  (It does not compile it.)
"""
import os
import posixpath
import re
import functools

def fnmatch(name, pat):
    """Test whether FILENAME matches PATTERN.

    Patterns are Unix shell style:

    *       matches everything
    ?       matches any single character
    [seq]   matches any character in seq
    [!seq]  matches any char not in seq

    An initial period in FILENAME is not special.
    Both FILENAME and PATTERN are first case-normalized
    if the operating system requires it.
    If you don't want this, use fnmatchcase(FILENAME, PATTERN).
    """
    name = os.path.normcase(name)
    pat = os.path.normcase(pat)
    return fnmatchcase(name, pat)

@functools.lru_cache(maxsize=32768, typed=True)
def _compile_pattern(pat):
    if isinstance(pat, bytes):
        pat_str = str(pat, 'ISO-8859-1')
        res_str = translate(pat_str)
        res = bytes(res_str, 'ISO-8859-1')
    else:
        res = translate(pat)
    return re.compile(res).match

def fnmatchcase(name, pat):
    """Test whether FILENAME matches PATTERN, including case.

    This is a version of fnmatch() which doesn't case-normalize
    its arguments.
    """
    match = _compile_pattern(pat)
    return match(name) is not None

def translate(pat):
    """Translate a shell PATTERN to a regular expression.

    There is no way to quote meta-characters.
    """

    STAR = object()
    parts = _translate(pat, STAR, '.')
    return _join_translated_parts(parts, STAR)


def _translate(pat, STAR, QUESTION_MARK):
    res = []
    add = res.append
    i, n = 0, len(pat)
    while i < n:
        c = pat[i]
        i = i+1
        if c == '*':
            # compress consecutive `*` into one
            if (not res) or res[-1] is not STAR:
                add(STAR)
        elif c == '?':
            add(QUESTION_MARK)
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j = j+1
            if j < n and pat[j] == ']':
                j = j+1
            while j < n and pat[j] != ']':
                j = j+1
            if j >= n:
                add('\\[')
            else:
                stuff = pat[i:j]
                if '-' not in stuff:
                    stuff = stuff.replace('\\', r'\\')
                else:
                    chunks = []
                    k = i+2 if pat[i] == '!' else i+1
                    while True:
                        k = pat.find('-', k, j)
                        if k < 0:
                            break
                        chunks.append(pat[i:k])
                        i = k+1
                        k = k+3
                    chunk = pat[i:j]
                    if chunk:
                        chunks.append(chunk)
                    else:
                        chunks[-1] += '-'
                    # Remove empty ranges -- invalid in RE.
                    for k in range(len(chunks)-1, 0, -1):
                        if chunks[k-1][-1] > chunks[k][0]:
                            chunks[k-1] = chunks[k-1][:-1] + chunks[k][1:]
                            del chunks[k]
                    # Escape backslashes and hyphens for set difference (--).
                    # Hyphens that create ranges shouldn't be escaped.
                    stuff = '-'.join(s.replace('\\', r'\\').replace('-', r'\-')
                                     for s in chunks)
                # Escape set operations (&&, ~~ and ||).
                stuff = re.sub(r'([&~|])', r'\\\1', stuff)
                i = j+1
                if not stuff:
                    # Empty range: never match.
                    add('(?!)')
                elif stuff == '!':
                    # Negated empty range: match any character.
                    add('.')
                else:
                    if stuff[0] == '!':
                        stuff = '^' + stuff[1:]
                    elif stuff[0] in ('^', '['):
                        stuff = '\\' + stuff
                    add(f'[{stuff}]')
        else:
            add(re.escape(c))
    assert i == n
    return res


def _join_translated_parts(inp, STAR):
    # Deal with STARs.
    res = []
    add = res.append
    i, n = 0, len(inp)
    # Fixed pieces at the start?
    while i < n and inp[i] is not STAR:
        add(inp[i])
        i += 1
    # Now deal with STAR fixed STAR fixed ...
    # For an interior `STAR fixed` pairing, we want to do a minimal
    # .*? match followed by `fixed`, with no possibility of backtracking.
    # Atomic groups ("(?>...)") allow us to spell that directly.
    # Note: people rely on the undocumented ability to join multiple
    # translate() results together via "|" to build large regexps matching
    # "one of many" shell patterns.
    while i < n:
        assert inp[i] is STAR
        i += 1
        if i == n:
            add(".*")
            break
        assert inp[i] is not STAR
        fixed = []
        while i < n and inp[i] is not STAR:
            fixed.append(inp[i])
            i += 1
        fixed = "".join(fixed)
        if i == n:
            add(".*")
            add(fixed)
        else:
            add(f"(?>.*?{fixed})")
    assert i == n
    res = "".join(res)
    return fr'(?s:{res})\Z'

ETERNALAI_MCP_PROXY_URL = os.getenv("ETERNALAI_MCP_PROXY_URL", None)
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in "true1yes"
PROXY_SCOPE: list[str] = os.getenv("PROXY_SCOPE", "*").split(',')

def need_redirect(url: str):
    return any(fnmatch(url, e) for e in PROXY_SCOPE)

def unpack_original_url(url: str, **kwargs):
    url_parts = parse.urlparse(url)

    mat = [
        e.split('=') 
        for e in url_parts.query.split('&')
    ]
    
    query = {
        **(kwargs.get('params') or {}),
        **{e[0]: e[1] for e in mat if len(e) == 2}
    }
    
    return {
        'url': f"{url_parts.scheme}://{url_parts.netloc}{url_parts.path}",
        'query': query
    }
    
def b64_encode_original_body(body: bytes):
    if body is None:
        return None

    return base64.b64encode(body).decode()

# Dictionary, list of tuples, bytes
def extract_body(**something) -> bytes:
    data = something.get('data')

    if data is not None:
        if isinstance(data, bytes):
            return data
        
        if isinstance(data, str):
            return data.encode()
        
        return json.dumps(data).encode()

    _json = something.pop('json')

    if json is not None:
        return json.dumps(_json).encode()

    return None

if ETERNALAI_MCP_PROXY_URL is not None:
    DEBUG_MODE and print("Start patching", file=sys.stderr)

    try:
        import http.client

        original_http_request = http.client.HTTPConnection.request
        def patch(self, method, url, body=None, headers=None): 
            if need_redirect(url):
                payload = {
                    "method": method, 
                    **unpack_original_url(url),
                    "body": b64_encode_original_body(body), 
                    "headers": headers
                }

                DEBUG_MODE and print("DEBUG-patching", payload, file=sys.stderr)

                res = original_http_request(
                    self, 'POST', ETERNALAI_MCP_PROXY_URL,
                    body=json.dumps(payload).encode(),
                    headers={
                        'Content-Type': 'application/json'
                    }
                )

                DEBUG_MODE and print("DEBUG-patching", res, file=sys.stderr)
            else:
                res = original_http_request(self, method, url, body=None, headers=None)

            
            return res
        
        http.client.HTTPConnection.request = patch
    except ImportError: pass

    try:
        import requests
        original_requests_session_request = requests.sessions.Session.request

        def patch(self, method, url, \
                    params=None, data=None, headers=None, cookies=None, files=None, \
                    auth=None, timeout=None, allow_redirects=True, proxies=None, \
                    hooks=None, stream=None, verify=None, cert=None, json=None):
            
            if need_redirect(url):
                payload = {
                    "method": method, 
                    **unpack_original_url(url), 
                    "body": b64_encode_original_body(extract_body(json=json, data=data)), 
                    "headers": headers or {}
                }
                
                DEBUG_MODE and print('DEBUG-patching', payload, file=sys.stderr)

                res = original_requests_session_request(
                    self, 'POST',  ETERNALAI_MCP_PROXY_URL,
                    json=payload,
                )
                
                DEBUG_MODE and print("DEBUG-patching", res.status_code, file=sys.stderr)

            else:
                res = original_requests_session_request(
                    self, method, url, \
                    params=None, data=None, headers=None, cookies=None, files=None, \
                    auth=None, timeout=None, allow_redirects=True, proxies=None, \
                    hooks=None, stream=None, verify=None, cert=None, json=None
                )

            return res

        requests.sessions.Session.request = patch

    except ImportError: pass

    try:
        import httpx
        original_httpx_client_send = httpx._client.Client.request
        
        def patch(self, method, url, \
            content, data, files, json, params, headers, cookies, \
            auth, follow_redirects, timeout, extensions): \
            
            if need_redirect(url):
                payload = {
                    "method": method, 
                    **unpack_original_url(url), 
                    "body": b64_encode_original_body(extract_body(json=json, data=data)), 
                    "headers": headers or {}
                }

                DEBUG_MODE and print("DEBUG-patching", payload, file=sys.stderr)
                
                res =  original_httpx_client_send(
                    self, 'POST', ETERNALAI_MCP_PROXY_URL, 
                    json={
                        "method": method, 
                        **unpack_original_url(url), 
                        "body": b64_encode_original_body(extract_body(json=json, data=data)), 
                        "headers": headers or {}
                    }
                )
                
                DEBUG_MODE and print("DEBUG-patching", res.status_code, file=sys.stderr)
            
            else:
                res =  original_httpx_client_send(self, method, url, \
                    content, data, files, json, params, headers, cookies, \
                    auth, follow_redirects, timeout, extensions
                )
            
            return res

            
        httpx._client.Client.request = patch

        original_httpx_async_client_send = httpx._client.AsyncClient.request
        use_client_default = httpx._client.UseClientDefault()

        async def override_httpx_async_client_send(self, method: str, url: str, content = None, data = None,
            files = None, json = None, params = None, headers = None, cookies = None, auth = use_client_default, 
            follow_redirects = use_client_default, timeout = use_client_default, extensions = None,
        ):
            
            if need_redirect(url):
                payload = {
                    "method": method, 
                    **unpack_original_url(url), 
                    "body": b64_encode_original_body(extract_body(json=json, data=data)), 
                    "headers": headers or {}
                }

                DEBUG_MODE and print("DEBUG-patching", payload, file=sys.stderr)
                
                res = await original_httpx_async_client_send(
                    self, 'POST', ETERNALAI_MCP_PROXY_URL, 
                    json=payload
                )
                
                DEBUG_MODE and print("DEBUG-patching", res.status_code, file=sys.stderr)
            
            else:
                res = await original_httpx_async_client_send(
                    self, method, url, content, data,
                    files, json, params, headers, cookies, auth, 
                    follow_redirects, timeout, extensions,
                )
            
            return res 

        httpx._client.AsyncClient.request = override_httpx_async_client_send
    except ImportError: pass

    try:
        import aiohttp
        original_aiohttp_client_request = aiohttp.client.request
        def patch(method, url, version, connector, loop, **kwargs):
            
            if need_redirect(url):
                
                payload = {
                    "method": method, 
                    **unpack_original_url(url), 
                    "body": b64_encode_original_body(extract_body(**kwargs)), 
                    "headers": kwargs.get('headers', {})
                }

                DEBUG_MODE and print("DEBUG-patching", payload, file=sys.stderr)
                
                res = original_aiohttp_client_request(
                    'POST', ETERNALAI_MCP_PROXY_URL,
                    json=payload
                )
                
                DEBUG_MODE and print("DEBUG-patching", res.status_code, file=sys.stderr)

            else:
                res = original_aiohttp_client_request(
                    method, url, version, connector, loop, **kwargs
                )
            
            return res 
            
        aiohttp.client.request = patch

        original_aiohttp_client_session_request = aiohttp.client.ClientSession._request
        async def override_aiohttp_client_session_request(
            self,
            method: str,
            str_or_url: str,
            *,
            params = None,
            data = None,
            json = None,
            cookies = None,
            headers = None,
            skip_auto_headers = None,
            auth = None,
            allow_redirects: bool = True,
            max_redirects: int = 10,
            compress = None,
            chunked = None,
            expect100 = False,
            raise_for_status = None,
            read_until_eof: bool = True,
            proxy = None,
            proxy_auth = None,
            timeout = aiohttp.helpers.sentinel,
            verify_ssl = None,
            fingerprint = None,
            ssl_context = None,
            ssl = True,
            server_hostname = None,
            proxy_headers = None,
            trace_request_ctx = None,
            read_bufsize = None,
            auto_decompress = None,
            max_line_size = None,
            max_field_size = None,
        ):
            if need_redirect(str_or_url):
                payload = {
                    "method": method, 
                    **unpack_original_url(str_or_url), 
                    "body": b64_encode_original_body(extract_body(json=json, data=data)), 
                    "headers": headers or {}
                }

                DEBUG_MODE and print("DEBUG-patching", payload, file=sys.stderr)
                
                res = await original_aiohttp_client_session_request(
                    self, 'POST', ETERNALAI_MCP_PROXY_URL,
                    json=payload,
                )
                
                DEBUG_MODE and print("DEBUG-patching", res.status_code, file=sys.stderr)
            
            else:
                res = await original_aiohttp_client_session_request(
                    self, method, str_or_url, params, data, json,
                    cookies, headers, skip_auto_headers, auth, allow_redirects,
                    max_redirects, compress, chunked, expect100, raise_for_status,
                    read_until_eof, proxy, proxy_auth, timeout, verify_ssl, fingerprint, 
                    ssl_context, ssl, server_hostname, proxy_headers, trace_request_ctx,
                    read_bufsize, auto_decompress, max_line_size, max_field_size,
                )
            
            return res
            
        aiohttp.client.ClientSession._request = override_aiohttp_client_session_request
    except ImportError: pass 
