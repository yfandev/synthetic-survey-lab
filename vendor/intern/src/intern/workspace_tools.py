"""Text filesystem tools that use the selected Executor, never host file I/O.

The backend image requires Python 3 and bash. This optional pack deliberately
handles text; image/document pipelines remain separate capabilities.
"""

from __future__ import annotations

import base64
import json
import shlex
from typing import ClassVar

from pydantic import BaseModel

from intern.executor import mount_of
from intern.tools.base import Effect, Tool, ToolContext, ToolOutput, fail
from intern.tools.files import EditTool, ReadTool, WriteTool
from intern.tools.search import FindTool, GrepTool
from intern.tools.shell import BashTool

_WORKER = r"""
import base64,fnmatch,hashlib,json,os,pathlib,re,sys,tempfile
request=json.loads(base64.b64decode(sys.argv[1]))
args=request['arguments'];operation=request['operation'];budget=request['budget']
mount=pathlib.Path(request['root']).resolve()
root=pathlib.Path.cwd().resolve()
def path(value):
    candidate=pathlib.Path(value).expanduser().resolve()
    if not candidate.is_relative_to(mount):
        raise ValueError('Filesystem tools are restricted to '+str(mount))
    return candidate
def emit(content):
    print(json.dumps({'status':'ok','content':content}))
def files(p):
    if p.is_file(): yield p;return
    for folder,dirs,names in os.walk(p,followlinks=False):
        dirs[:]=sorted(d for d in dirs if d not in {'.git','node_modules','.venv','__pycache__'})
        for name in sorted(names):
            child=(pathlib.Path(folder)/name).resolve()
            if child.is_relative_to(mount): yield child
try:
    p=path(args.get('path','.'))
    if operation=='read':
        if p.is_dir(): emit('\n'.join(x.name+('/' if x.is_dir() else '') for x in sorted(p.iterdir())[:500]))
        else:
            if p.stat().st_size>10_000_000: raise ValueError('Text file exceeds 10 MB')
            data=p.read_bytes()
            if b'\0' in data[:8192]: raise ValueError('Binary file: use a format-specific tool')
            lines=data.decode(errors='replace').splitlines();start=args.get('offset',1)-1
            if start>max(0,len(lines)-1): raise ValueError('Offset is past the end of the file')
            stop=min(len(lines),start+args.get('limit',len(lines))) if args.get('limit') else len(lines)
            selected=[];size=0
            for line in lines[start:stop]:
                if selected and size+len(line)+1>max(120,budget-120): break
                selected.append(line);size+=len(line)+1
            text='\n'.join(selected);end=start+len(selected)
            if end<len(lines): text+='\n[Continue with offset='+str(end+1)+'.]'
            emit(text or '(empty file)')
    elif operation in ('write','edit'):
        if operation=='write': text=args['content']
        else:
            text=p.read_bytes().decode('utf-8',errors='surrogateescape')
            if not args['old_string']: raise ValueError('old_string must not be empty')
            count=text.count(args['old_string'])
            if not count: raise ValueError('old_string was not found')
            if count>1 and not args.get('replace_all'):
                raise ValueError('old_string is ambiguous; narrow it or set replace_all')
            text=text.replace(args['old_string'],args['new_string'],-1 if args.get('replace_all') else 1)
        p.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(dir=p.parent,prefix='.intern-write-')
        try:
            with os.fdopen(fd,'wb') as f: f.write(text.encode('utf-8',errors='surrogateescape'))
            if p.exists(): os.chmod(temp,p.stat().st_mode)
            os.replace(temp,p)
        finally:
            if os.path.exists(temp): os.unlink(temp)
        emit('Saved '+str(p.relative_to(root)))
    elif operation in ('grep','find'):
        rows=[];regex=re.compile(args['pattern'],re.I if args.get('ignore_case') else 0) if operation=='grep' else None
        cap=args.get('max_matches',args.get('limit',1000))
        for item in files(p):
            relative=str(item.relative_to(root))
            if operation=='find':
                if fnmatch.fnmatch(relative,args['pattern']) or fnmatch.fnmatch(item.name,args['pattern']):
                    rows.append(relative)
            elif (not args.get('glob') or fnmatch.fnmatch(relative,args['glob'])
                  or fnmatch.fnmatch(item.name,args['glob'])):
                if item.stat().st_size>10_000_000: continue
                data=item.read_bytes()
                if b'\0' in data[:8192]: continue
                for line_no,line in enumerate(data.decode(errors='replace').splitlines(),1):
                    if regex.search(line): rows.append(relative+':'+str(line_no)+':'+line[:300])
                    if len(rows)>=cap: break
            if len(rows)>=cap or sum(len(x)+1 for x in rows)>budget: break
        emit('\n'.join(rows) or 'No matches.')
except Exception as e:
    print(json.dumps({'status':'error','content':'Error: '+str(e)+
        '\nNext: correct the path/arguments or use bash for format-specific work.'}))
"""


def _backend_tool(original: Tool) -> Tool:
    class WorkspaceTool(Tool):
        name: ClassVar[str] = original.name
        description: ClassVar[str] = original.description + " Uses the selected workspace executor; text files only."
        Args: ClassVar[type[BaseModel]] = original.Args
        effects: ClassVar[frozenset[Effect]] = original.effects

        async def run(self, args: BaseModel, ctx: ToolContext) -> ToolOutput:
            mount = mount_of(ctx.executor)
            if mount is None:
                return fail("This file tool needs a sandboxed executor.", "Use the standard file tools instead.")
            payload = base64.b64encode(
                json.dumps(
                    {
                        "root": str(mount.inner),
                        "operation": self.name,
                        "arguments": args.model_dump(mode="json"),
                        "budget": ctx.max_output_chars,
                    }
                ).encode()
            ).decode()
            if len(payload) > 100_000:
                return fail("File operation payload is too large for this backend.", "Write smaller pieces using bash.")
            result = await ctx.executor.run(
                shlex.join(["python3", "-c", _WORKER, payload]),
                cwd=ctx.cwd,
                timeout_s=60,
                max_output_bytes=max(ctx.max_output_chars * 8, 16_384),
                abort=ctx.abort,
            )
            if result.aborted or result.timed_out or result.exit_code != 0 or result.omitted_bytes:
                return fail(
                    "Workspace operation did not complete reliably.", "Check the executor and narrow the operation."
                )
            try:
                return ToolOutput.model_validate_json(result.output)
            except ValueError:
                return fail("Workspace returned invalid output.", "Check that the backend has Python 3 available.")

    return WorkspaceTool()


def workspace_tools() -> list[Tool]:
    """Replace all host filesystem tools together; pair with an executor that gives a `mount` (Docker, or a
    sandbox backend)."""
    return [*[_backend_tool(t) for t in [ReadTool(), WriteTool(), EditTool(), GrepTool(), FindTool()]], BashTool()]
