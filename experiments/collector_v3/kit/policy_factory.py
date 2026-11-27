"""Single explicit dispatch for live, causal replay and local research."""
from model import Policy


def make_policy(settings, *args, **kwargs):
    if settings.mode == 'article_pd':
        from article_policy import ArticlePolicy
        cls = ArticlePolicy
    elif settings.memory != 'none' or settings.feedback_rule != 'legacy':
        from memory_policy import MemoryPolicy
        cls = MemoryPolicy
    else:
        cls = Policy
    return cls(settings, *args, **kwargs)
