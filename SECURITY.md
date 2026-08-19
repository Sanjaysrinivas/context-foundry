# Security policy

## Supported versions

Security fixes are applied to the latest release on `main`.

## Reporting

Please report a suspected vulnerability through GitHub's private vulnerability reporting for this repository. Do not include sensitive documents, API keys, or exploit details in a public issue.

## Deployment boundary

Local RAG is a single-user portfolio application. It binds to `127.0.0.1`, has no authentication, and is not designed to be exposed directly to the internet or an untrusted local network.

Before any shared deployment, add and verify:

- authenticated access and authorization;
- TLS at the network boundary;
- per-user document isolation;
- rate and upload limits enforced by the serving layer;
- malware scanning and hardened document parsing;
- secrets management and scrubbed logs;
- a secured remote Qdrant deployment with backups.

The default Ollama configuration keeps inference local. Selecting an OpenAI-compatible remote URL sends prompts or document chunks to that endpoint; its privacy and retention policy then applies.

Retrieved documents are untrusted input. The application includes prompt-injection guidance, but model instructions are not a security boundary. Never grant model output authority to execute commands or perform external actions without separate validation and authorization.

