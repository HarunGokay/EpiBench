"""EpiBench: Epistemic long-term memory benchmark."""

__version__ = "1.0.0"

import sys
# Support both epibench and EpiBench module access
sys.modules.setdefault("EpiBench", sys.modules[__name__])
