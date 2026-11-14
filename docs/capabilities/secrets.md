# Scoped secret references

Applications can wrap an existing executor in `SecretExecutor`. Configure an
`EnvironmentSecrets` resolver with named `EnvironmentBinding`s, authorized scopes
and source environment names; configure a fixed executor scope and target variable
names mapped to `SecretRef`s. Credentials resolve immediately before dispatch.
Unknown/missing/denied references fail before the subprocess starts. There is no
model-facing secret lookup tool and no credential value in a tool schema.

```python
from intern.executor import LocalExecutor
from intern.secrets import EnvironmentBinding, EnvironmentSecrets, SecretExecutor, SecretRef

ref = SecretRef(name="search-provider")
resolver = EnvironmentSecrets(
    [EnvironmentBinding(reference=ref, source_env="BRAVE_SEARCH_API_KEY", scopes=frozenset({"research"}))]
)
session.deps.executor = SecretExecutor(
    LocalExecutor(),
    resolver,
    scope="research",
    inject={"SEARCH_API_KEY": ref},
)
```

Only the configured base environment allowlist is inherited; caller environment
overrides cannot replace injected names or introduce other variables. Literal
credential values are redacted from returned executor output, and exception
details from credential-bearing boundaries are withheld. Scope and bindings
belong to application code, not model arguments. Custom resolvers implement the
same `SecretResolver` protocol, whose `resolve` is `async` (a resolver may run a
command, query a database or check access on every call); returned values use
Pydantic `SecretStr`. `CommandSecrets` serves values a host command prints, and
`RoutedSecrets` serves each name from the resolver it is configured in.

This is not an exfiltration sandbox. Authorized code can read its injected
credentials, transform them, send them over a network or write them to files;
literal output redaction cannot prevent these actions. Use narrow scopes,
appropriate executor isolation and egress rules, and redact exported artifacts
and telemetry separately. Default environment allowlists intentionally exclude
provider keys and shell initialization variables. No defaults or prompts change.
