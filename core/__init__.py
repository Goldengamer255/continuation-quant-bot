"""Shared infrastructure used by both research and live execution.

Keeping configuration and logging here (instead of duplicating them in
``research/`` and ``execution/``) means there is exactly one definition of
every strategy parameter. Research and live trading cannot drift apart.
"""
