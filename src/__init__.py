"""
Microplastic Analyzer - Source Package
"""

__version__ = '2.0.0'
__author__ = 'Microplastic Research Team'

# Import main modules for convenience
from . import core
from . import analysis
from . import data_generation

__all__ = [
    'core',
    'analysis',
    'data_generation',
]
