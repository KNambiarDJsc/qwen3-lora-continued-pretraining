"""Dataset adapters — one per dataset source."""
from slm_research.data.adapters.wikitext import WikitextAdapter
from slm_research.data.adapters.openwebtext import OpenWebTextAdapter
from slm_research.data.adapters.bookcorpusopen import BookCorpusOpenAdapter
from slm_research.data.adapters.tinystories import TinyStoriesAdapter
from slm_research.data.adapters.ag_news import AgNewsAdapter
from slm_research.data.adapters.xsum import XSumAdapter
from slm_research.data.adapters.cnn_dailymail import CnnDailymailAdapter
from slm_research.data.adapters.daily_dialog import DailyDialogAdapter
from slm_research.data.adapters.eli5 import ELI5Adapter
from slm_research.data.adapters.fineweb_edu import FineWebEduAdapter

__all__ = [
    "WikitextAdapter",
    "OpenWebTextAdapter",
    "BookCorpusOpenAdapter",
    "TinyStoriesAdapter",
    "AgNewsAdapter",
    "XSumAdapter",
    "CnnDailymailAdapter",
    "DailyDialogAdapter",
    "ELI5Adapter",
    "FineWebEduAdapter",
]
