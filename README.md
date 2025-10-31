# Agent4Vul

## Requirements

## Setups

### Openhands Agent

The Openhands SDK is cloned from https://github.com/OpenHands/software-agent-sdk.
To install the SDK:

```
cd agents/openhands_agent-sdk
make build
```

To run the experiments with Openhands, firtly add the `agents/experiments` folder into the `agents/openhands_agent-sdk` folder and run the command `uv`:

```
cd agents
cp -r experiments openhands_agent-sdk
uv run python experiments/main.py
```
