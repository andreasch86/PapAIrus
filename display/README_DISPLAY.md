## Prerequisites

After generating documentation for a repository with PapAIrus, change into the `display` folder:

```bash
cd display
```

You need **Node.js 10**. The provided script uses nvm to install it.

## One-command deployment scripts

Run `make help` to view the available automation targets:

```bash
make help
```

Use `make init_env` to install nvm and Node.js 10 automatically, or install Node.js 10 manually if you prefer. On Windows, run the command prompt as an administrator.

Then run `make init` once to initialise the GitBook environment. After the environment is ready, you can re-run `make serve` whenever you change configuration or `book.json` to redeploy the book.

A successful run looks like:

```
init!
finish!
info: >> generation finished with success in 16.7s !

Starting server ...
Serving book on http://localhost:4000
```

Open http://localhost:4000/ to view your rendered GitBook documentation.

## Future TODO List

- [✅] One-click environment creation
- [ ] (If local setup is difficult) Docker-based one-click GitBook deployment and upload
- [ ] One-click deployment to the relevant GitHub or Gitee Pages site for direct access via the repository URL
