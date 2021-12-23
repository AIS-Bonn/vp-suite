from typing import List
from datetime import datetime
import argparse

def most(l: List[bool], factor=0.67):
    '''
    Like List.all(), but not 'all' of them.
    '''
    return sum(l) >= factor * len(l)

def timestamp(program):
    """ Obtains a timestamp of the current system time in a human-readable way """

    timestamp = str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-")
    return f"{program}_{timestamp}"

class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = float(v)
        setattr(namespace, self.dest, my_dict)