from app.services.rate_limit import SlidingWindowRateLimiter


def test_rate_limiter_allows_limit_then_rejects_next_request() -> None:
    now = [100.0]
    limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60.0, clock=lambda: now[0])

    assert limiter.allow("client") is True
    assert limiter.allow("client") is True
    assert limiter.allow("client") is False


def test_rate_limiter_expires_old_requests() -> None:
    now = [100.0]
    limiter = SlidingWindowRateLimiter(max_requests=1, window_seconds=60.0, clock=lambda: now[0])

    assert limiter.allow("client") is True
    now[0] = 161.0
    assert limiter.allow("client") is True
