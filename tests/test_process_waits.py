"""A job never hangs on input it was not asked to take. That a user's message never waits behind the
agent waiting on a job is the hand-over's now: tests/test_harness_waits.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from intern.processes import ProcessManager


async def test_a_job_that_reads_its_input_never_waits_on_it_unless_asked_to(tmp_path: Path):
    # `docker exec -i` jobs and Django's test runner hung until killed. A job reads
    # end-of-file on its input now, as a bash call does, unless it was started to take input.
    async with ProcessManager() as manager:
        job = await manager.start("cat; echo done", cwd=tmp_path)
        assert (await manager.poll(job.handle, wait_s=3)).output == "done\n"
        with pytest.raises(ValueError, match="stdin: true"):
            await manager.write(job.handle, "x")
        fed = await manager.start("cat; echo done", cwd=tmp_path, stdin=True)
        assert (await manager.poll(fed.handle, wait_s=0.5)).state == "running"
        await manager.write(fed.handle, "typed\n", eof=True)
        assert (await manager.poll(fed.handle, wait_s=3)).output == "typed\ndone\n"
