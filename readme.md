# G1-Deploy

## Installation

```bash
pip install "mujoco<3.4"
# at g1-deploy/
pip install -e . --no-build-isolation
# at cursor workspace folder
pybind11-stubgen g1_deploy -o stubs
```

In your cursor/vscode workspace, create and modify `.vscode/settings.json`:

```json
{
    // for cursor
    "cursorpyright.analysis.extraPaths": [
        "${workspaceFolder}/stubs",
    ]
}
```

or for vscode
```json
{
    "python.analysis.extraPaths" : [
        "${workspaceFolder}/stubs",
    ]
}
```

## State Machine

Startup: built-in control

* build-in-control --> user control (L1 + R1)
* user control --> damping mode (L2 + B)
* damping mode --> build-in-control (L1 + R1)
