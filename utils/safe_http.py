from __future__ import annotations

import ipaddress
import json
import socket
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    HTTPHandler,
    OpenerDirector,
    Request,
    build_opener,
)


BLOCKED_LITERAL_HOSTS = frozenset({"localhost", "broadcasthost", "ip6-localhost", "ip6-loopback"})


def _is_public_ip(ip: ipaddress._BaseAddress) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolved_ips(host: str) -> list[ipaddress._BaseAddress]:
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return []
    ips: list[ipaddress._BaseAddress] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            ips.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return ips


def is_safe_article_url(url: str) -> bool:
    """URLの構文だけで弾ける不正ホストを拒否する。DNS解決は行わない。"""
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"}:
        return False
    host = parsed.hostname
    if not host:
        return False
    if host.lower() in BLOCKED_LITERAL_HOSTS:
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return True
    return _is_public_ip(address)


def is_public_resolved_host(url: str) -> bool:
    """DNS 解決して、いずれかのIPが内部レンジならFalse。すべて公開IPなら True。"""
    if not is_safe_article_url(url):
        return False
    parsed = urlparse(url)
    host = parsed.hostname or ""
    try:
        ipaddress.ip_address(host)
        return True  # IPリテラルの時点で is_safe_article_url 側で検証済み
    except ValueError:
        pass

    ips = _resolved_ips(host)
    if not ips:
        return False
    return all(_is_public_ip(ip) for ip in ips)


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if not is_public_resolved_host(newurl):
            raise HTTPError(newurl, code, f"安全でないリダイレクト先: {newurl}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _build_safe_opener() -> OpenerDirector:
    return build_opener(HTTPHandler(), HTTPSHandler(), _SafeRedirectHandler())


def safe_urlopen(url: str, *, timeout: int = 20, user_agent: str = "Type-2/1.0") -> Any:
    if not is_public_resolved_host(url):
        raise ValueError(f"安全でないURLを拒否しました: {url}")
    opener = _build_safe_opener()
    req = Request(url, headers={"User-Agent": user_agent})
    return opener.open(req, timeout=timeout)


def safe_fetch_text(
    url: str,
    *,
    timeout: int = 20,
    max_bytes: int = 1_000_000,
    user_agent: str = "Type-2/1.0",
) -> str:
    with safe_urlopen(url, timeout=timeout, user_agent=user_agent) as response:
        raw = response.read(max_bytes)
        encoding = response.headers.get_content_charset() or "utf-8"
    return raw.decode(encoding, errors="replace")


def safe_fetch_json(
    url: str,
    *,
    timeout: int = 20,
    max_bytes: int = 500_000,
    user_agent: str = "Type-2/1.0",
) -> Any:
    """SSRF保護付きでURLからJSONを取得・デコードする。"""
    with safe_urlopen(url, timeout=timeout, user_agent=user_agent) as response:
        raw = response.read(max_bytes)
    return json.loads(raw.decode("utf-8"))
