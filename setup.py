import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import List
from setuptools import find_packages, setup

here = Path(__file__).parent
long_description = (here / 'README.md').read_text()

def _load_py_module(fname, pkg="vp_suite"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(str(here), pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py

about = _load_py_module("__about__.py")

def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    """Load requirements from a file.
    >>> _load_requirements(str(here))  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http") or "@http" in ln:
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
setup(
    name=about.__name__,
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/Flunzmas/vp-suite",
    license=about.__license__,
    packages=find_packages(exclude=["tests*"]),
    package_data={'vp_suite': ['resources/*']},
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI", "video prediction"],
    python_requires=">=3.8",
    setup_requires=[],
    install_requires=_load_requirements(str(here)),
    project_urls={
        "Documentation": about.__docs_url__,
        "Source": about.__source_url__,
        "Tracker": about.__tracker_url__,
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
