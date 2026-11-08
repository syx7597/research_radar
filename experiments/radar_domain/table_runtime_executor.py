"""Runtime-only normalization of invalid model operation configurations.

The frozen executor and source data stay unchanged. Valid operations delegate
unchanged. Only three parameter-error classes become ordinary ValueError tool
feedback, which the existing TableEpisode already records and can replay.
Infrastructure failures (including OSError and MemoryError) propagate.
"""
from decimal import DecimalException

from .table_program import TableExecutor


class TableRuntimeExecutor(TableExecutor):
    def operate(self, function, config, dependencies):
        try:
            return super().operate(function, config, dependencies)
        except (DecimalException, AttributeError, OverflowError) as exc:
            raise ValueError("invalid_operation_configuration:" + type(exc).__name__) from exc
