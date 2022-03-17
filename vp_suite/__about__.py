import time

_this_year = time.strftime("%Y")
__name__ = "vp-suite"
_version = 0, 0, 9
__version__ = ".".join(map(str, _version))
__author__ = "Andreas Boltres"
__author_email__ = "andreas.boltres@posteo.de"
__license__ = "MIT"
__copyright_short__ = f"2021-{_this_year}, {__author__}"
__copyright__ = f"Copyright (c) {__copyright_short__}."
__homepage__ = "https://ais-bonn.github.io/vp-suite/"
__docs_url__ = "https://ais-bonn.github.io/vp-suite/"
__source_url__ = "https://github.com/AIS-Bonn/vp-suite"
__tracker_url__ = "https://github.com/AIS-Bonn/vp-suite/issues"
__docs__ = ( "A Framework for Training and Evaluating Video Prediction Models" )
__python_requires__ = ">=3.6"
__keywords__ = ["deep learning", "pytorch", "AI", "video prediction"]
__classifiers__ = [
        "Environment :: Console",
        "Natural Language :: English",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
    ]

__all__ = ["__name__", "__version__", "__author__", "__author_email__", "__license__",
           "__copyright__", "__homepage__", "__docs_url__", "__source_url__",
           "__tracker_url__", "__docs__", "__keywords__", "__classifiers__",
           "__python_requires__"]
