import sys
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from git import Repo

test_impl_filepaths = [path for path in Path(__file__).parent.rglob("_*.py") if "__init__" not in str(path)]
test_impl_filepath_stems = ["impl_" + p.stem[1:] for p in test_impl_filepaths]


class add_path:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


@pytest.mark.parametrize('impl_file_path', test_impl_filepaths, ids=test_impl_filepath_stems)
@pytest.mark.slow
def test_module_implementations_against_references(impl_file_path: Path):
    spec = importlib.util.spec_from_file_location(impl_file_path.stem, impl_file_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    with TemporaryDirectory() as repo_dir:
        _ = Repo.clone_from(test_module.REFERENCE_GIT_URL, repo_dir)
        with add_path(repo_dir):
            test_module.test_impl()
