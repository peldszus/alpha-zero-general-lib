import pytest
import ray


@pytest.fixture
def local_ray():
    ray.init(local_mode=True, ignore_reinit_error=True)
    yield
    ray.shutdown()
