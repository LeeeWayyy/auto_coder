# Supervisor Examples

Practical examples demonstrating Supervisor workflows and features.

## Available Examples

| Example | Description | Difficulty |
|---------|-------------|------------|
| [basic-workflow](basic-workflow/) | Simple plan → implement → review workflow | Beginner |
| [custom-gates](custom-gates/) | Creating custom verification gates | Intermediate |
| [multi-model-routing](multi-model-routing/) | Adaptive model selection | Intermediate |
| [parallel-workflow](parallel-workflow/) | Parallel component execution | Advanced |
| [workflows](workflows/) | Declarative graph workflows (Studio) | Intermediate |

## Quick Start

Each example directory contains:
- `README.md`: Detailed instructions
- `.supervisor/`: Pre-configured Supervisor setup
- Sample files and scripts

To run an example:

```bash
cd examples/basic-workflow
supervisor plan "Implement greeting functionality"
```

## Learning Path

1. **Start with [basic-workflow](basic-workflow/)**: Learn the fundamentals
2. **Try [custom-gates](custom-gates/)**: Add project-specific verification
3. **Experiment with [multi-model-routing](multi-model-routing/)**: Optimize model selection
4. **Scale with [parallel-workflow](parallel-workflow/)**: Maximize throughput

## Contributing Examples

Have a useful example? Please contribute!

1. Create a new directory under `examples/`
2. Add README.md with clear instructions
3. Include `.supervisor/` configuration
4. Add to this index
5. Submit a pull request

---

For more information, see the [Getting Started Guide](../docs/GETTING_STARTED.md) and the [Runbook](../runbook/README.md).
