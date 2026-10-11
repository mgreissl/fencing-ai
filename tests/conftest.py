"""Pytest configuration for fencing-ai tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: slow end-to-end variants (deselect with -m 'not slow')"
    )
