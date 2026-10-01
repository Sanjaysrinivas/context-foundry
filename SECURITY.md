# Security policy

## Supported versions

Security fixes are applied to the latest release on `main`.

## Reporting

Please report a suspected vulnerability through GitHub's private vulnerability reporting for this repository. Do not include sensitive documents, API keys, or exploit details in a public issue.

## Deployment boundary

Context Foundry is a single-user portfolio application. It binds to `127.0.0.1`, has no authentication, and is not designed to be exposed directly to the internet or an untrusted local network.

Browser responses set a restrictive same-origin content policy, deny framing and MIME sniffing, suppress referrer data, and mark API responses as non-cacheable. Embedded Qdrant operations are serialized inside the process, but multiple application processes must not share the same local database path.

Generated answers are rendered as Markdown with raw HTML disabled. The browser only inserts the server-rendered `answer_html`; API consumers that render the plain `answer` field must apply equivalent restrictions instead of trusting model output as HTML.

Original uploads are retained beneath `RAG_DATA_DIR/sources/<collection-hash>` so the local UI can
download them and render PDF pages. Stored filenames are metadata and never become filesystem
destinations. Replacing or deleting a document removes its retained source; clearing the collection
removes all retained sources for that collection. Source downloads and page previews are served
with `Cache-Control: no-store`, but any user or process with access to the data directory can read
the files, so protect that directory and include it in the same backup and deletion policy as the
vector store.

Before any shared deployment, add and verify:

- authenticated access and authorization;
- TLS at the network boundary;
- per-user document isolation;
- rate and upload limits enforced by the serving layer;
- malware scanning and hardened document parsing;
- secrets management and scrubbed logs;
- a secured remote Qdrant deployment with backups.

The default Ollama configuration keeps inference local. Selecting an OpenAI-compatible remote URL sends prompts or document chunks to that endpoint; its privacy and retention policy then applies.

`RAG_JEV_MODE=observe` independently enables TypeSafe's hosted Jev API. It sends generated claims
and the text of their actual cited excerpts, including sensitive content if it occurs in those
excerpts. Observation excludes uncited passages from the request state. `RAG_JEV_MODE=repair`
also submits alternative passages already retrieved for the same question part, individually,
when looking for replacement citations. Claim text and the retrieved ledger are preserved;
citations change only after a fresh support check passes the threshold. Filenames, credentials,
retained original files, and passages retrieved only for other question parts remain excluded from
each check.
Keep `TYPESAFE_API_KEY` in the Git-ignored `.env` or environment. Audit records expose sanitized
status errors rather than remote error bodies. They are rendered as plain text in the browser.
Observation does not filter answers. Citation repair does not rewrite claims or establish that
the answer is true, complete, or relevant; Jev judgments are not a security boundary.

Retrieved documents are untrusted input. The application includes prompt-injection guidance, but model instructions are not a security boundary. Never grant model output authority to execute commands or perform external actions without separate validation and authorization.

