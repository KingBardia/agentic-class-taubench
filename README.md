# Tau-Bench Airline Assessment with Turn Efficiency

A modified Tau-Bench assessment focused on **airline domain tasks** with a custom **turn efficiency metric**. Built for AgentBeats v2 remote mode.

## Custom Modifications

1. **Airline Domain Focus**: Filters tasks to airline-only scenarios
2. **Turn Efficiency Metric**: Tracks `turns` - the number of interaction turns between green and white agents per task
3. **Concise Response Prompt**: White agent is instructed to minimize turns for efficiency

## Metrics Reported

- `success`: Boolean indicating task completion (reward == 1)
- `turns`: Number of interaction turns taken to complete the task
- `time_used`: Total time for task completion

## Project Structure

```
src/
├── green_agent/    # Assessment manager (tracks turns, reports metrics)
├── white_agent/    # Target agent (with efficiency prompt)
└── launcher.py     # Evaluation coordinator
```

## Installation

```bash
# Prerequisites (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y python3 python3-pip

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone <your-repo-url>
cd agentic-class-taubench
uv sync
source .venv/bin/activate
```

## Local Testing

```bash
# Configure API key (uses Claude Haiku)
echo "ANTHROPIC_API_KEY=your-key" > .env

# Launch complete evaluation
uv run python main.py launch
```

## AgentBeats v2 Remote Mode

### Running with Controller

```bash
# Terminal 1: Green agent
HTTPS_ENABLED=true CLOUDRUN_HOST=your-domain.trycloudflare.com ROLE=green agentbeats run_ctrl

# Terminal 2: White agent
HTTPS_ENABLED=true CLOUDRUN_HOST=your-domain.trycloudflare.com ROLE=white agentbeats run_ctrl
```

### Register on v2.agentbeats.org

1. Go to [v2.agentbeats.org](https://v2.agentbeats.org)
2. **Add Agent**:
   - Name: `tau-airline-efficiency`
   - Type: `remote`
   - Controller URL: Your cloudflared URL
   - Agent URL: Agent endpoint
   - Git branch: (optional)
3. **Verify**: Reload page, check agent card appears
4. **Create Assessment**:
   - Select green agent
   - Add white agent(s)
   - Set repeat count (e.g., 1)
   - Config: default
5. **Monitor**: Check audit logs with auto-refresh
6. **Share**: Use copy link feature

## Example Config

```json
{
  "env": "airline",
  "user_strategy": "llm",
  "user_model": "claude-3-haiku-20240307",
  "user_provider": "anthropic",
  "task_split": "test",
  "task_ids": [1]
}
```

## Bug Reports & Support

- GitHub Issues: [Open an issue](https://github.com/agentbeats/agentify-example-tau-bench/issues)
- GitHub Discussions: [Start a discussion](https://github.com/agentbeats/agentify-example-tau-bench/discussions)
- Security: sec+agentbeats@berkeley.edu

## License

See original [tau-bench](https://github.com/sierra-research/tau-bench) for licensing.
