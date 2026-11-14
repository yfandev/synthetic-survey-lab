# Optional browser automation

Install the optional browser extra and prepare Chromium explicitly:

```sh
uv sync --extra browser
uv run --extra browser playwright install chromium
```

Create one `BrowserSession(artifact_dir, BrowserOptions(...))` per application
session, use its async context (or `aclose`) and append `browser_tool(browser)` to
the selected tool list. The tool exposes navigate/read/screenshot/click/fill and
bounded DOM-text pagination. Screenshots use a fixed viewport and byte limit,
remain in the caller's artifact directory with content hashes, and are attached
as images only for a vision-capable tool context. There is no process-global pack
or inherited browser profile. No browser/network activity occurs on import.
The browser process receives a small environment allowlist rather than ambient
provider API keys or shell-initialization variables.
Screenshot publication uses the reader's private-file boundary: mode0600 files in an owner-private
mode0700 directory, no followed store/target symlinks and no overwrite of different existing bytes.
Existing broad-permission stores are refused. This requires POSIX permissions and does not isolate
other processes running as the same user.

Every session owns a fresh Chromium process, nonpersistent context and an
authenticated loopback forward proxy. The proxy opens upstream sockets using the
web reader's `PublicNetworkBackend`: validate every DNS answer and connect only to
the approved numeric address. Chromium uses the fixed proxy with implicit
loopback bypass disabled, QUIC disabled, nonproxied WebRTC UDP disabled and direct
DNS resolution disabled. Redirects and subresources use the same socket boundary;
there is no preflight-DNS-then-browser-resolution race. Default destination ports
are 80/443 and private/nonpublic addresses are refused. WebRTC/WebTransport page
constructors are disabled before page scripts run. WebSockets and service
workers are blocked, downloads disabled and extra pages closed.

`allow_private_urls=True` is an explicit application/operator setting for local
development; it is never a model argument. It broadens the proxy's destination
scope, including ports. The proxy is private to the service, requires random
credentials and has connection/byte/time bounds. Closing the session closes its
browser, proxy listener and tunnels. An interrupted browser action is not marked
safe to retry because a click may already have changed a remote application.

This is egress control for an unmodified configured Chromium process, not a
guarantee against a compromised browser/kernel. Chromium's own sandbox is enabled;
high-risk deployments should additionally apply OS/container network policy and
resource isolation. Web content is untrusted data; DOM text does not become an
instruction. The application still wraps the browser Tool with its normal effect
policy/receipts and task authorization. Browser restart does not restore cookies
or prove remote-action rollback. Default tools/profiles are unchanged. This service reuses the
merged web-reader network and private-file boundaries; paired eval evidence is required
before enabling it by default.

Run real local Chromium tests with `INTERN_BROWSER_TEST=1`. Tests cover synthetic
form actions and retained screenshots, fresh cookie contexts, private-address
denial, redirects/subresources through the validated numeric connector, in-flight abort cleanup
and private screenshot storage with unsafe-target refusal.
