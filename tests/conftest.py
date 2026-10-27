import pytest

from jobpilot import db


@pytest.fixture
def conn():
    c = db.connect(":memory:")
    yield c
    c.close()
