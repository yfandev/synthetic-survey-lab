# Docker workspace execution

Applications can opt into `DockerExecutor(workspace, DockerOptions(image=...))`
and replace the complete base tool list with `workspace_tools()`. Both shell and
all five text filesystem tools then use the same executor. Merely replacing the
shell executor while leaving host file tools enabled is not workspace isolation.

```python
from intern.runners.docker import DockerExecutor, DockerOptions
from intern.workspace_tools import workspace_tools

session.deps.executor = DockerExecutor(workspace, DockerOptions(image="python:3.12.11-slim-bookworm"))
session.deps.tools = workspace_tools()
```

Prepare the image explicitly; execution uses `--pull never`. Use a digest for
repeatable deployments. The image requires bash and Python 3. Each command runs in
a new container with a read-only root, dropped capabilities, resource limits,
no-new-privileges and no network by default. Only the selected workspace is mounted
read/write at `/workspace`; files persist there, while package installs and
container-local state do not persist. Network `bridge` requires an explicit
operator choice. Cancellation/deadlines remove the uniquely owned container even
when stopping the Docker client alone would leave it running.

Filesystem tools resolve paths inside `/workspace`, reject escapes and handle
bounded text reads/search. They intentionally do not duplicate image/document
conversion. The host working directory must be inside the configured mount.
Credentials are not interpolated into command arguments; explicit environment
values can be forwarded by name. The Docker daemon and chosen image are trusted;
this does not protect against a malicious daemon or image or isolate concurrent
agents sharing the same writable workspace. Create a dedicated workspace per run.

No daemon probe, pull or container starts at import. Default profiles are unchanged.
Run the real integration test with `INTERN_DOCKER_IMAGE` set to an already prepared
image. Paired task evals remain required before recommending default activation.
