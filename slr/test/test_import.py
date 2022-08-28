import pytest
from sys import platform


@pytest.mark.skipif(platform != 'win32', reason="test only on windows")
def test_import_submodule():
    try:
        import slr
        import slr.data
        import slr.utils
        import slr.static
        import slr.data.asl
        import slr.model
    except Exception as e:
        raise ModuleNotFoundError
        