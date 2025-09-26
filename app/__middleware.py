import os
import json
from urllib import parse
import base64
import sys
from fnmatch import fnmatch

import logging
logger = logging.getLogger(__name__)

AGENT_NAME = os.getenv("AGENT_NAME")
ETERNALAI_MCP_PROXY_URL = os.getenv("ETERNALAI_MCP_PROXY_URL", "http://localhost:33030/84532-proxy/prompt") 
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in "true1yes"
PROXY_SCOPE: list[str] = os.getenv("PROXY_SCOPE", "*").split(',')

def need_redirect(url: str):
    return any(fnmatch(url, e) for e in PROXY_SCOPE)

def construct_v2_payload(method, url, body, headers, params: dict={}) -> dict:
    is_json_encodeable_body = True

    try:
        if isinstance(body, bytes):
            body = body.decode()

        if isinstance(body, str):
            body = json.loads(body)

        json.dumps(body)
    except (TypeError, json.JSONDecodeError):
        is_json_encodeable_body = False
        DEBUG_MODE and print("DEBUG-construct_v2_payload", body, file=sys.stderr)

    payload_content = {
        "method": method, 
        **unpack_original_url(url, params=params),
        "headers": headers
    }
    
    if method.upper() == 'POST' or method.upper() == 'PUT':
        payload_content['body'] = body if is_json_encodeable_body else '{}'

    return {
        'messages': [
            {
                'role': 'user',
                'content': json.dumps(payload_content)
            }
        ]
    }

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
        import requests
        original_requests_session_request = requests.sessions.Session.request

        def patch(self, method, url, \
                    params=None, data=None, headers=None, cookies=None, files=None, \
                    auth=None, timeout=None, allow_redirects=True, proxies=None, \
                    hooks=None, stream=None, verify=None, cert=None, json=None):
            
            
            if need_redirect(url):
                headers = headers or {}
                if AGENT_NAME:
                    headers['eai-source'] = AGENT_NAME

                body = extract_body(json=json, data=data)
                payload = construct_v2_payload(method, url, body, headers or {}, params=params or {})
                DEBUG_MODE and print('DEBUG-patching', payload, file=sys.stderr)

                res = original_requests_session_request(
                    self, 'POST', ETERNALAI_MCP_PROXY_URL,
                    json=payload,
                    headers={
                        'Content-Type': 'application/json'
                    },
                )
                
                DEBUG_MODE and print("DEBUG-patching", res.status_code, file=sys.stderr)

            else:
                res = original_requests_session_request(
                    self, method, url, \
                    params=params, data=data, headers=headers, cookies=cookies, files=files, \
                    auth=auth, timeout=timeout, allow_redirects=allow_redirects, proxies=proxies, \
                    hooks=hooks, stream=stream, verify=verify, cert=cert, json=json
                )

            return res

        requests.sessions.Session.request = patch

    except ImportError: pass
