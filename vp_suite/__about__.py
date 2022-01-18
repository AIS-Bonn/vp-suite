import time

_this_year = time.strftime("%Y")
__name__ = "vp-suite"
_version = 0, 0, 6
__version__ = ".".join(map(str, _version))
__author__ = "Andreas Boltres"
__author_email__ = "andreas.boltres@posteo.de"
__license__ = "MIT"
__copyright_short__ = f"2021-{_this_year}, {__author__}"
__copyright__ = f"Copyright (c) {__copyright_short__}."
__homepage__ = "https://github.com/Flunzmas/vp-suite"
__docs_url__ = "https://flunzmas-vp-suite.readthedocs.io/en/latest/"
__source_url__ = "https://github.com/Flunzmas/vp-suite"
__tracker_url__ = "https://github.com/Flunzmas/vp-suite/issues"
__docs__ = ( "A Framework for Training and Evaluating Video Prediction Models" )

__all__ = ["__name__", "__author__", "__author_email__", "__copyright__", "__docs__",
           "__homepage__", "__license__", "__version__"]
