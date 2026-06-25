# Security Policy

This is a research prototype for local MIDI experimentation. It is not hardened as a network-facing production service.

## Reporting

Please report security issues privately to the repository owner through GitHub.

## Notes

- Do not expose the local FastAPI/WebSocket interface directly to the public internet.
- Treat listening-study endpoints and spreadsheet deployment URLs as secrets unless they are explicitly public test endpoints.
- Do not commit Hugging Face tokens, API keys, Google Apps Script deployment secrets, or private dataset material.
