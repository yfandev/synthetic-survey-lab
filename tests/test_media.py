"""Images: read shows them to a model that can see, and each provider sends them where its API accepts them."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from PIL import Image

from intern.executor import LocalExecutor
from intern.hooks import Extension
from intern.messages import ImagePart, Message, ToolResult, ToolResultMessage
from intern.models.profile import Profile
from intern.models.registry import Endpoint
from intern.providers.anthropic import AnthropicProvider
from intern.providers.base import ProviderRequest
from intern.providers.fake import Turn
from intern.providers.openai_compat import OpenAICompatProvider
from intern.tools import images
from intern.tools.base import Tool, ToolContext, ToolOutput
from intern.tools.files import ReadArgs, ReadTool


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(
        cwd=tmp_path,
        executor=LocalExecutor(),
        abort=asyncio.Event(),
        max_output_chars=20_000,
        image_dir=tmp_path / "images",
    )


async def read(ctx: ToolContext, **arguments) -> ToolOutput:
    tool = ReadTool()
    parsed = tool.parse(arguments)
    assert isinstance(parsed, ReadArgs), parsed
    return await tool.run(parsed, ctx)


async def test_an_image_is_attached_resized_for_a_model_that_can_see(ctx: ToolContext):
    Image.new("RGB", (4000, 3000), "white").save(ctx.cwd / "chart.png")
    out = await read(ctx, path="chart.png")
    assert out.status == "ok" and len(out.images) == 1
    image = out.images[0]
    assert max(image.width, image.height) <= images.MAX_EDGE
    assert image.width * image.height <= images.MAX_PIXELS
    assert image.source == "chart.png" and image.path.endswith(".png")


async def test_workspace_read_keeps_attaching_images_with_its_existing_constructor(ctx: ToolContext):
    from intern.extensions.workspace.extension import Workspace

    tool = next(tool for tool in Workspace().tools() if isinstance(tool, ReadTool))
    Image.new("RGB", (50, 50), "white").save(ctx.cwd / "scan.png")
    out = await tool.run(ReadArgs(path="scan.png"), ctx)
    assert out.status == "ok" and len(out.images) == 1


async def test_a_model_that_cannot_see_gets_the_usual_binary_file_answer(ctx: ToolContext):
    Image.new("RGB", (50, 50), "white").save(ctx.cwd / "scan.png")
    ctx.image_dir = None
    out = await read(ctx, path="scan.png")
    assert out.status == "error" and not out.images and "binary file" in out.content


@pytest.mark.parametrize("suffix", [*sorted(images.IMAGE_SUFFIXES), ".PNG"])
@pytest.mark.parametrize("vision", [True, False])
async def test_disabled_image_attachment_does_not_load_or_store_bytes(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch, suffix: str, vision: bool
):
    path = ctx.cwd / f"chart{suffix}"
    path.touch()
    store = ctx.image_dir
    if not vision:
        ctx.image_dir = None

    def unexpected(*args, **kwargs):
        pytest.fail("read must not load image bytes when attachments are disabled")

    monkeypatch.setattr(Path, "read_bytes", unexpected)
    monkeypatch.setattr(images, "load_image", unexpected)
    out = await ReadTool(attach_images=False).run(ReadArgs(path=path.name), ctx)
    assert out.status == "ok" and not out.images and not out.resources
    assert path.name in out.content and "image" in out.content
    assert store is not None and not store.exists()


async def test_disabled_image_attachment_preserves_text_paging_and_directory_reads(ctx: ToolContext):
    (ctx.cwd / "notes.txt").write_text("first\nsecond\nthird\n")
    (ctx.cwd / "assets.png").mkdir()
    tool = ReadTool(attach_images=False)
    page = await tool.run(ReadArgs(path="notes.txt", offset=2, limit=1), ctx)
    assert page.status == "ok" and page.content.startswith("second\n") and "offset=3" in page.content
    assert len(page.resources) == 1
    directory = await tool.run(ReadArgs(path="."), ctx)
    assert directory.status == "ok" and "assets.png/" in directory.content and "notes.txt" in directory.content
    empty = await tool.run(ReadArgs(path="assets.png"), ctx)
    assert empty.status == "ok" and "empty directory" in empty.content
    missing = await tool.run(ReadArgs(path="missing.png"), ctx)
    assert missing.status == "error" and not missing.images


async def test_host_extension_disables_read_images_but_its_view_tool_can_still_attach(make_harness):
    admitted: list[ImagePart] = []

    class ViewImages(ReadTool):
        name = "view_images"

        async def run(self, args: ReadArgs, ctx: ToolContext) -> ToolOutput:
            return ToolOutput(content="Host-admitted image", images=admitted)

    class HostImages(Extension):
        def tools(self) -> list[Tool]:
            return [ReadTool(attach_images=False), ViewImages()]

    harness = make_harness(
        [
            Turn(tool_calls=[("read", {"path": "chart.png"})]),
            Turn(tool_calls=[("view_images", {"path": "chart.png"})]),
            Turn(text="The host's image is visible."),
        ],
        profile=Profile(vision=True),
        extensions=[HostImages()],
    )
    Image.new("RGB", (10, 10), "black").save(harness.cwd / "chart.png")
    admitted.append(image_part(harness.cwd, "admitted", "white"))
    outcome = await harness.run()
    assert outcome.reason == "completed"
    requests = harness.provider.requests
    after_read = [m.result for m in requests[1].messages if isinstance(m, ToolResultMessage)]
    assert len(after_read) == 1 and after_read[0].status == "ok" and not after_read[0].images
    assert not (harness.deps.spill_dir / "images").exists()
    after_view = [m.result for m in requests[-1].messages if isinstance(m, ToolResultMessage)]
    assert [len(result.images) for result in after_view] == [0, 1]
    assert after_view[-1].images == admitted
    for provider in (anthropic(), OpenAICompatProvider(Endpoint(provider="x", model="m", base_url="http://x"))):
        wire = json.dumps(provider.body(requests[1]))
        assert '"image"' not in wire and '"image_url"' not in wire
        wire = json.dumps(provider.body(requests[-1]))
        assert wire.count('"type": "image"') + wire.count('"type": "image_url"') == 1


def tool_result(call_id: str, parts: list[ImagePart]) -> ToolResultMessage:
    return ToolResultMessage(
        result=ToolResult(call_id=call_id, tool_name="read", status="ok", content="shown", images=parts)
    )


def image_part(tmp_path: Path, name: str, color: str = "white") -> ImagePart:
    path = tmp_path / f"{name}.png"
    Image.new("RGB", (10, 10), color).save(path)
    return ImagePart(path=str(path), mime="image/png", width=10, height=10, source=name, sha256=name)


def anthropic() -> AnthropicProvider:
    return AnthropicProvider(Endpoint(provider="anthropic", model="m", base_url="http://x", api="anthropic"))


def test_the_native_api_gets_images_inside_the_tool_result(tmp_path: Path):
    messages: list[Message] = [tool_result("a", [image_part(tmp_path, "one")]), tool_result("b", [])]
    body = anthropic().body(ProviderRequest(model="m", messages=messages, max_output_tokens=10))
    first, second = body["messages"][-1]["content"]
    assert [b["type"] for b in first["content"]] == ["text", "image"]
    assert first["content"][1]["source"]["media_type"] == "image/png"
    assert second["content"] == "shown"


def test_the_native_api_reports_a_missing_image_file(tmp_path: Path):
    gone = ImagePart(path=str(tmp_path / "gone.png"), mime="image/png", width=1, height=1, source="gone", sha256="0")
    body = anthropic().body(ProviderRequest(model="m", messages=[tool_result("a", [gone])], max_output_tokens=10))
    assert "no longer available" in json.dumps(body["messages"][-1])


def test_chat_completions_get_images_in_a_user_message_after_the_tool_results(tmp_path: Path):
    provider = OpenAICompatProvider(Endpoint(provider="x", model="m", base_url="http://x"))
    gone = ImagePart(path=str(tmp_path / "gone.png"), mime="image/png", width=1, height=1, source="gone", sha256="0")
    messages: list[Message] = [tool_result("a", [image_part(tmp_path, "one")]), tool_result("b", [gone])]
    wire = provider.body(ProviderRequest(model="m", messages=messages, max_output_tokens=10))["messages"]
    assert [m["role"] for m in wire] == ["tool", "tool", "user"]
    parts = wire[-1]["content"]
    assert [p["type"] for p in parts].count("image_url") == 1
    assert parts[-1] == {"type": "text", "text": "[gone: the image file is no longer available]"}
