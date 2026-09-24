"""Polite HTTP GET: at most one request per MIN_INTERVAL_S, exponential-backoff retries."""

import time

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

MIN_INTERVAL_S = 0.5
MAX_ATTEMPTS = 5
USER_AGENT = "EuroHoops-Analytics/0.1 (+https://github.com/straf10/EuroHoops-Analytics)"


def make_client() -> httpx.Client:  # pragma: no cover - real network transport
    return httpx.Client(timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.TransportError)


class Fetcher:
    """Wraps an httpx client; every request (including retries) respects the throttle."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._last_request = float("-inf")

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(MAX_ATTEMPTS),
        wait=wait_exponential(multiplier=1, max=30),
        reraise=True,
    )
    def get(self, url: str) -> bytes:
        wait = self._last_request + MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()
        response = self._client.get(url)
        response.raise_for_status()
        return response.content
