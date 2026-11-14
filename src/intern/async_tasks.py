"""Keep owned asynchronous work inside its caller's lifetime boundary."""

from __future__ import annotations

import asyncio


async def join_tasks(*tasks: asyncio.Task[object], cancel: bool = False) -> None:
    """Settle owned children, even if the waiter is cancelled repeatedly.

    Retrieve child errors without replacing the caller's outcome. Cancellation
    received during cleanup is re-raised only after every child has finished.
    Already-cancelling children are joined without interrupting cleanup again.
    A child that suppresses cancellation forever still requires host intervention.
    """
    if cancel:
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
    joined = asyncio.gather(*tasks, return_exceptions=True)
    cancelled = False
    while not joined.done():
        try:
            await asyncio.shield(joined)
        except asyncio.CancelledError:
            cancelled = True
    joined.result()
    if cancelled:
        raise asyncio.CancelledError
