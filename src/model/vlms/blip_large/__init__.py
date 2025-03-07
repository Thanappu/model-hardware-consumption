from ._base import *

def get_model(format='base'):
    if format == 'base':
        import _base
        return _base.VlmsBlipLarge

