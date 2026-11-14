# Host-controlled image viewing

Requires **Intern 0.28.0 or a commit containing this change**. Upgrade an older package or commit
pin before passing the new constructor argument. The package is `intern-agent`, the command
`intern`, and the Python import package `intern`.

By default, the built-in `read` tool attaches image files when the model supports vision.
A host that admits images through its own viewing tool can disable that behavior:

```python
from intern.tools.files import ReadTool

read = ReadTool(attach_images=False)
```

For an existing image path, this instance returns a successful text notice saying the file is
an image and directing the agent to the host's image-viewing tool. It does not read, decode,
resize, store or attach the image bytes. Detection uses the same filename suffixes as the normal
image path, case-insensitively: PNG, JPG/JPEG, GIF, WebP, BMP and TIF/TIFF. Missing paths still
return an error; text paging and directory listings keep working.

## Install through an extension

The existing `Extension.tools()` mechanism replaces a pack tool with a tool of the same name.
For a CLI or JSON-RPC host, put this in an importable module, for example `intern_images.py`:

```python
from intern.hooks import Extension
from intern.tools.base import Tool
from intern.tools.files import ReadTool


class HostImages(Extension):
    def tools(self) -> list[Tool]:
        return [ReadTool(attach_images=False)]
```

Then load it with `intern rpc --extension intern_images:HostImages` (plus your normal model
and other options). The module must be on Python's import path. If the host already supplies
an extension, return the configured `ReadTool` from that extension's `tools()` instead. When
several extensions replace `read`, the last replacement wins.

This example changes the local base reader; the host must also provide its own viewing tool.
Keep the Workspace capability disabled:

```toml
[capabilities.workspace]
enabled = false
```

Workspace loads after fixed extensions and supplies its own document-aware `read`. If it would
replace a fixed extension's image-disabled reader, session setup refuses the configuration with
an error naming this setting. Reordering `--extension` arguments cannot fix that conflict. This
check prevents a capability from silently undoing the opt-out; it does not configure Workspace's
reader or remove its document support.

Keep vision enabled for the model. This setting controls only this `read` instance; it does not
filter other tools' images or provider requests. The host should describe its viewing tool
(for example, a `view_images` tool) in its instructions and keep enforcing image admission in its broker.
With the disabled reader selected, Intern's vision instructions point to the available viewing
tools instead of recommending `read` for images. The default reader keeps its existing instructions.

Images already in conversation history remain, both on ordinary follow-ups and in resumed
sessions. A host with admission limited to the current turn must retain replay filtering or
readmit those images before sending later requests. This option removes new image attachments
from the configured reader; it does not replace that history handling or govern browser/MCP images.

[Docker and OpenSandbox runners](capabilities/runners.md) use separate, text-only filesystem tools.
A local `ReadTool` replacement is not a compatible adapter for those runners; setup rejects it.
Retain their filesystem tools when using them.

Hosts upgrading from the old `nace-harness` package must also follow the
[Intern rename migration](acp.md#0250), including command, import, environment and home names.

## Why a setting

The host already owns its viewing tool, so disabling this image producer reuses the existing tool
replacement mechanism. A new image-loader hook would add an asynchronous admission and error
contract. Filtering tool results alone happens after loading and preparing bytes; broker filtering
still serves the separate purpose of enforcing admission for every image in a request.
