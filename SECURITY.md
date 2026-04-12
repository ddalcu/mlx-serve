# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest release | Yes |
| Older releases | No |

## Reporting a Vulnerability

If you discover a security vulnerability in mlx-serve, please report it responsibly:

1. **Do not** open a public issue
2. Email **security@dalcu.com** with:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

You should receive a response within 48 hours. We will work with you to understand and address the issue before any public disclosure.

## Security Model

mlx-serve is designed as a **local development tool** running on a single machine. It is not designed for production deployment or untrusted network exposure.

### By Design
- **No authentication**: The HTTP API has no auth — it's intended for localhost use only
- **Agent tool execution**: The agent mode executes shell commands and file operations based on model output. This is inherently powerful and should only be used with trusted models
- **Workspace confinement**: File tools (readFile, writeFile, editFile, searchFiles, listFiles) are confined to the working directory. Shell commands are not confined

### Recommendations
- Bind to `127.0.0.1` (default) — do not expose to the network
- Only load models from trusted sources
- Review agent actions in the MLX Core chat UI
- Do not run the server as root
