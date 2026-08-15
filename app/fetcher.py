"""Guarded, pinned-address HTTP fetching for explicit deep URL analysis."""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import ipaddress
import socket
import ssl
import zlib
from urllib.parse import urljoin, urlsplit


ALLOWED_PORTS = {80, 443}
MAX_REDIRECTS = 3
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_URL_LENGTH = 2_048
SOCKET_TIMEOUT_SECONDS = 5.0
USER_AGENT = "PhiUSIIL-Research-Analyser/2.0"


class FetchError(RuntimeError):
    """Safe client-facing failure from guarded webpage retrieval."""


@dataclass(frozen=True)
class FetchResult:
    """Bounded HTML response used by feature extraction."""

    final_url: str
    body: bytes
    content_type: str
    redirects: int


def _validated_addresses(hostname: str, port: int) -> list[str]:
    try:
        records = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise FetchError("Hostname could not be resolved.") from error
    addresses = sorted({record[4][0] for record in records})
    if not addresses:
        raise FetchError("Hostname did not resolve to an address.")
    parsed_addresses = [ipaddress.ip_address(address) for address in addresses]
    if any(not address.is_global for address in parsed_addresses):
        raise FetchError("Private, local, reserved, or non-global addresses are blocked.")
    return addresses


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, pinned_ip: str) -> None:
        super().__init__(host, port=port, timeout=SOCKET_TIMEOUT_SECONDS)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout
        )


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, pinned_ip: str) -> None:
        super().__init__(
            host,
            port=port,
            timeout=SOCKET_TIMEOUT_SECONDS,
            context=ssl.create_default_context(),
        )
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        raw_socket = socket.create_connection(
            (self._pinned_ip, self.port), timeout=self.timeout
        )
        self.sock = self._context.wrap_socket(raw_socket, server_hostname=self.host)


def _request_once(url: str) -> tuple[int, dict[str, str], bytes]:
    if len(url) > MAX_URL_LENGTH:
        raise FetchError("URL exceeds the analysis length limit.")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise FetchError("Only absolute HTTP and HTTPS URLs can be fetched.")
    if parsed.username is not None or parsed.password is not None:
        raise FetchError("URLs containing credentials are blocked.")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as error:
        raise FetchError("URL port is invalid.") from error
    if port not in ALLOWED_PORTS:
        raise FetchError("Only standard HTTP and HTTPS ports are allowed.")

    addresses = _validated_addresses(parsed.hostname, port)
    target = parsed.path or "/"
    if parsed.query:
        target += "?" + parsed.query
    host_header = parsed.hostname
    if port not in {80, 443}:
        host_header += f":{port}"
    last_error: Exception | None = None
    for address in addresses[:2]:
        connection: http.client.HTTPConnection
        if parsed.scheme == "https":
            connection = _PinnedHTTPSConnection(parsed.hostname, port, address)
        else:
            connection = _PinnedHTTPConnection(parsed.hostname, port, address)
        try:
            connection.request(
                "GET",
                target,
                headers={
                    "Accept": "text/html,application/xhtml+xml;q=0.9",
                    "Accept-Encoding": "identity",
                    "Connection": "close",
                    "Host": host_header,
                    "User-Agent": USER_AGENT,
                },
            )
            response = connection.getresponse()
            headers = {key.casefold(): value for key, value in response.getheaders()}
            declared_length = headers.get("content-length")
            if declared_length and declared_length.isdecimal() and int(declared_length) > MAX_RESPONSE_BYTES:
                raise FetchError("Webpage response is larger than the analysis limit.")
            body = response.read(MAX_RESPONSE_BYTES + 1)
            if len(body) > MAX_RESPONSE_BYTES:
                raise FetchError("Webpage response is larger than the analysis limit.")
            return response.status, headers, body
        except FetchError:
            raise
        except (OSError, ssl.SSLError, http.client.HTTPException) as error:
            last_error = error
        finally:
            connection.close()
    raise FetchError("Webpage could not be fetched securely.") from last_error


def fetch_html(url: str) -> FetchResult:
    """Fetch bounded HTML with revalidation and IP pinning at every redirect."""
    current_url = url
    for redirect_count in range(MAX_REDIRECTS + 1):
        status, headers, body = _request_once(current_url)
        if status in {301, 302, 303, 307, 308}:
            location = headers.get("location")
            if not location:
                raise FetchError("Redirect response did not include a destination.")
            if redirect_count == MAX_REDIRECTS:
                raise FetchError("Webpage exceeded the redirect limit.")
            current_url = urljoin(current_url, location)
            continue
        if status < 200 or status >= 400:
            raise FetchError(f"Webpage returned HTTP status {status}.")
        content_type = headers.get("content-type", "").split(";", 1)[0].strip().casefold()
        if content_type not in {"text/html", "application/xhtml+xml"}:
            raise FetchError("Webpage response is not HTML.")
        content_encoding = headers.get("content-encoding", "identity").casefold()
        if content_encoding in {"gzip", "x-gzip"}:
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
            try:
                body = decompressor.decompress(body, MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES or decompressor.unconsumed_tail:
                    raise FetchError("Decompressed webpage is larger than the analysis limit.")
                body += decompressor.flush(MAX_RESPONSE_BYTES + 1 - len(body))
            except zlib.error as error:
                raise FetchError("Compressed webpage response is invalid.") from error
            if len(body) > MAX_RESPONSE_BYTES or decompressor.unconsumed_tail:
                raise FetchError("Decompressed webpage is larger than the analysis limit.")
        elif content_encoding not in {"", "identity"}:
            raise FetchError("Webpage compression format is not supported.")
        return FetchResult(
            final_url=current_url,
            body=body,
            content_type=content_type,
            redirects=redirect_count,
        )
    raise FetchError("Webpage exceeded the redirect limit.")
