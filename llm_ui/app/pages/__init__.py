"""
Pages module for the UPCAST Profiler application.
This module contains various UI pages.
"""

# Import all page modules here
# Keep these imports for compatibility with older code
from . import home
from . import configure
from . import provider
try:
    from . import consumer_QA
except ImportError:
    pass  # Consumer_QA is optional