
from unittest.main import MODULE_EXAMPLES

from slr.utils.utils import only_test_on_windows


@only_test_on_windows
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
        