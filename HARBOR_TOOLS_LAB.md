# Harbor Metals · Section 1 lab

1. Start with [01_architectures.ipynb](notebooks/01_architectures.ipynb): fixed RAG, fixed RAG plus calculator, then agent-selected calls. The default model is scripted, explicitly identified, and runs offline.
2. Continue to [01b_harbor_tools_mcp_permissions.ipynb](notebooks/01b_harbor_tools_mcp_permissions.ipynb): the same company, direct OpenAI function calling, shared tools through MCP, and permission checks.

## Browser

- [Open Section 1 in Colab](https://colab.research.google.com/github/mfrdixon/STAC_AI_Bootcamp/blob/main/notebooks/01_architectures.ipynb).
- [Open MCP/permissions companion in Colab](https://colab.research.google.com/github/mfrdixon/STAC_AI_Bootcamp/blob/main/notebooks/01b_harbor_tools_mcp_permissions.ipynb).

Alternatively use Colab **File → Open notebook → GitHub**, paste the GitHub notebook URL, then **Save a copy in Drive**. Run the helper-file setup first.

## Local

Clone the repository and launch JupyterLab from `notebooks/`. Keep both helper modules and the MCP server script alongside the notebooks. For optional integrations run:

```bash
python -m pip install -r requirements-harbor-tools.txt
```

Default cells need no API key. Live GPT calls use `OPENAI_API_KEY` or a hidden prompt. The first notebook can also use Together. Never save keys in cells or outputs. Only fictional company information is sent.

## What is verified

- Offline walkthrough: initial reported-cash calculation 3.00x; eligible-cash calculation 3.25x; agent follows annex.
- Permission checks: research cannot release; reviewer can release; revoked reviewer cannot release. Release is simulated.
- Actual local MCP v1.30.0: initialize, discover three tools, retrieve annex, calculate 3.25x, deny research release, permit reviewer release.
- Live Together calls reached retrieval tools, but tested Llama/Qwen runs skipped the calculator. A Qwen run also produced an incorrect formula and ratio. The notebook flags missing calculator execution; these runs are not successful end-to-end task validations.
- OpenAI live calls require an account key; no key was available during preparation, so this provider path is not claimed as live-tested.

## Boundaries

The retriever is a small keyword demonstration, not production semantic retrieval. Empty results do not prove a document is absent. No model's successful behavior is guaranteed. MCP standardizes discovery and calls; it does not create permissions. Trusted launcher configuration supplies local demo identities. Production requires authenticated identities, server-side policy and durable records. The notebook checks authority to call a tool, not the truth or release suitability of a brief.
