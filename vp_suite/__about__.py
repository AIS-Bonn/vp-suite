import time

_this_year = time.strftime("%Y")
__name__ = "vp-suite"
__version__ = "0.0.2"
__author__ = "Andreas Boltres"
__author_email__ = "andreas.boltres@posteo.de"
__license__ = "MIT"
__copyright__ = f"Copyright (c) 2022-{_this_year}, {__author__}."
__homepage__ = "https://github.com/Flunzmas/vp-suite"
#__docs_url__ = "NONE"
__source_url__ = "https://github.com/Flunzmas/vp-suite"
__tracker_url__ = "https://github.com/Flunzmas/vp-suite/issues"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = ( "A Framework for Training and Evaluating Video Prediction Models" )

__all__ = ["__name__", "__author__", "__author_email__", "__copyright__", "__docs__",
           "__homepage__", "__license__", "__version__"]
