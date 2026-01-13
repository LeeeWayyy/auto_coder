# Operations Guide

Production deployment, monitoring, and maintenance guide for Supervisor.

## Table of Contents

1. [Deployment](#deployment)
2. [Monitoring](#monitoring)
3. [Troubleshooting](#troubleshooting)
4. [Maintenance](#maintenance)
5. [Production Best Practices](#production-best-practices)
6. [Scaling](#scaling)
7. [Disaster Recovery](#disaster-recovery)

---

## Deployment

### Requirements

**System Requirements**:
- **OS**: Linux (Ubuntu 20.04+, RHEL 8+), macOS 12+
- **Python**: 3.11 or 3.12
- **Docker**: 24.0.0+ (required for sandbox isolation)
- **Git**: 2.30+ (for worktree management)
- **Disk Space**: 10GB+ (for Docker images and worktrees)
- **Memory**: 4GB+ RAM (8GB+ recommended for parallel execution)

**Network Requirements**:
- Outbound HTTPS to AI API endpoints:
  - `api.anthropic.com` (Claude)
  - `api.openai.com` (OpenAI/Codex)
  - Google AI endpoints (Gemini)
- Docker network configuration for egress control (recommended)

### Installation

#### 1. Install Python Dependencies

```bash
# Clone repository
git clone https://github.com/your-org/supervisor.git
cd supervisor

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install Supervisor
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

#### 2. Install Docker

**Ubuntu/Debian**:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker ps
```

**macOS**:
```bash
# Install Docker Desktop
brew install --cask docker

# Start Docker Desktop
open -a Docker

# Verify installation
docker --version
docker ps
```

#### 3. Configure Network Security (Recommended)

For production, configure Docker network egress allowlist:

```bash
# Create egress-controlled network
docker network create \
  --driver bridge \
  --opt com.docker.network.bridge.name=br-supervisor \
  supervisor-network

# Configure iptables rules (requires root)
sudo iptables -I FORWARD -i br-supervisor -j DROP
sudo iptables -I FORWARD -i br-supervisor -d api.anthropic.com -j ACCEPT
sudo iptables -I FORWARD -i br-supervisor -d api.openai.com -j ACCEPT

# Make rules persistent (Ubuntu)
sudo apt-get install iptables-persistent
sudo netfilter-persistent save
```

#### 4. Configure API Keys

Set environment variables for AI APIs:

```bash
# Add to ~/.bashrc or ~/.zshrc
export ANTHROPIC_API_KEY="your-claude-api-key"
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"

# Or use a .env file (recommended for production)
cat > .env <<EOF
ANTHROPIC_API_KEY=your-claude-api-key
OPENAI_API_KEY=your-openai-api-key
GEMINI_API_KEY=your-gemini-api-key
EOF

# Load environment
source .env
```

#### 5. Initialize Project

```bash
# Initialize in your repository
cd /path/to/your/project
supervisor init

# Verify installation
supervisor --version
supervisor roles
```

### Docker Image Management

**Pull required images**:
```bash
# Python base image (for AI CLI and gates)
docker pull python:3.11-slim

# Custom images (if needed)
docker pull your-org/custom-ai-image:latest
```

**Build custom image** (optional):
```bash
# Dockerfile for custom image
cat > Dockerfile.supervisor <<EOF
FROM python:3.11-slim

# Install common tools
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install AI CLIs
RUN pip install --no-cache-dir \\
    anthropic-claude-cli \\
    openai-codex

WORKDIR /workspace
EOF

# Build
docker build -f Dockerfile.supervisor -t supervisor-runtime:latest .
```

**Update configuration to use custom image**:
```yaml
# .supervisor/config.yaml
sandbox:
  image: supervisor-runtime:latest
```

---

## Monitoring

### Metrics Dashboard

View performance metrics:

```bash
# Last 7 days
supervisor metrics --days 7

# Last 30 days (default)
supervisor metrics

# Continuous monitoring (when implemented)
supervisor metrics --live
```

**Metrics Tracked**:
- Execution count by role
- Success/failure rates
- Average duration
- Retry counts
- Token usage
- Model performance

### Database Monitoring

Monitor database size and performance:

```bash
# Check database size
ls -lh .supervisor/state.db

# Query event count
sqlite3 .supervisor/state.db "SELECT COUNT(*) FROM events;"

# Check workflow status
sqlite3 .supervisor/state.db \
  "SELECT workflow_id, status, timestamp FROM workflows ORDER BY timestamp DESC LIMIT 10;"
```

### Log Monitoring

Configure logging level:

```bash
# Set log level
export SUPERVISOR_LOG_LEVEL=INFO

# Available levels: DEBUG, INFO, WARNING, ERROR
```

**Log locations**:
- Stdout/stderr: Real-time command output
- `.supervisor/logs/` (if configured): Persistent logs
- Docker container logs: `docker logs <container-id>`

### Performance Monitoring

**Key Performance Indicators (KPIs)**:

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Success Rate | >90% | 80-90% | <80% |
| Avg Duration (implementer) | <5s | 5-10s | >10s |
| Retry Rate | <5% | 5-15% | >15% |
| Gate Pass Rate | >95% | 90-95% | <90% |
| Disk Usage | <50GB | 50-100GB | >100GB |

**Monitor with cron**:
```bash
# Add to crontab
0 */6 * * * cd /path/to/project && supervisor metrics --days 1 >> /var/log/supervisor-metrics.log
```

### Alerting

**Set up alerts for critical issues**:

```bash
#!/bin/bash
# /usr/local/bin/supervisor-health-check.sh

cd /path/to/project

# Check success rate
SUCCESS_RATE=$(supervisor metrics --days 1 | grep "Success Rate" | awk '{print $3}' | tr -d '%')

if (( $(echo "$SUCCESS_RATE < 80" | bc -l) )); then
    echo "ALERT: Success rate dropped to ${SUCCESS_RATE}%"
    # Send notification (email, Slack, PagerDuty, etc.)
    curl -X POST https://hooks.slack.com/your-webhook \
      -d "{\"text\": \"Supervisor success rate: ${SUCCESS_RATE}%\"}"
fi

# Check disk usage
DISK_USAGE=$(du -sh .supervisor | awk '{print $1}')
echo "Disk usage: ${DISK_USAGE}"

# Check for failed workflows
FAILED_COUNT=$(sqlite3 .supervisor/state.db \
  "SELECT COUNT(*) FROM workflows WHERE status='failed' AND timestamp > datetime('now', '-1 day');")

if [ "$FAILED_COUNT" -gt 5 ]; then
    echo "ALERT: ${FAILED_COUNT} workflows failed in last 24 hours"
fi
```

```bash
# Add to crontab (every hour)
0 * * * * /usr/local/bin/supervisor-health-check.sh
```

---

## Troubleshooting

### Common Issues

#### Docker Not Available

**Symptoms**:
```
Error: Docker is required but not available
```

**Diagnosis**:
```bash
# Check Docker status
docker ps
systemctl status docker  # Linux
```

**Solutions**:
```bash
# Start Docker daemon (Linux)
sudo systemctl start docker
sudo systemctl enable docker

# Restart Docker Desktop (macOS)
killall Docker && open -a Docker

# Check permissions
groups  # Should include 'docker'
sudo usermod -aG docker $USER
newgrp docker
```

#### Network Egress Not Configured

**Symptoms**:
```
WARNING: Cannot verify egress rules (permission denied)
```

**Diagnosis**:
```bash
# Check network rules (requires sudo)
sudo iptables -L FORWARD -n -v | grep br-supervisor
```

**Solutions**:
```bash
# Configure egress rules (see Deployment section)
# Or disable checking (not recommended for production):
export SUPERVISOR_SKIP_EGRESS_CHECK=1
```

#### Gate Failures

**Symptoms**:
```
✗ test failed (exit code 1)
  3 tests failed in tests/test_auth.py
```

**Diagnosis**:
```bash
# Check gate output
supervisor status --workflow-id wf-abc123

# Run gate manually in worktree
cd .supervisor/.worktrees/wf-abc123-step-001
pytest -v
ruff check .
```

**Solutions**:
```bash
# Fix failing tests
# Update gate configuration if needed:
# .supervisor/gates.yaml
gates:
  test:
    timeout: 600  # Increase timeout
    fail_action: warn  # Change to warning instead of error
```

#### Out of Memory

**Symptoms**:
```
Docker container killed (exit code 137)
```

**Diagnosis**:
```bash
# Check memory usage
free -h
docker stats

# Check Docker memory limits
docker info | grep -i memory
```

**Solutions**:
```bash
# Increase Docker memory limit (macOS: Docker Desktop preferences)

# Configure container limits:
# .supervisor/config.yaml
sandbox:
  memory_limit: 4g
  cpu_limit: 2.0

# Reduce parallel execution
workflow:
  parallel_execution: false
```

#### Slow Performance

**Symptoms**:
- Commands take >30 seconds
- High CPU usage
- Frequent retries

**Diagnosis**:
```bash
# Check metrics
supervisor metrics --days 1

# Profile specific operations
time supervisor run implementer "Test task"

# Check system resources
top
iotop  # I/O usage
```

**Solutions**:
```bash
# Enable caching
# .supervisor/gates.yaml
gates:
  test:
    cache: true

# Use faster models for simple tasks
# .supervisor/config.yaml
roles:
  implementer:
    cli: claude:haiku  # Faster model

# Reduce context size
# .supervisor/roles/implementer.yaml
context:
  token_budget: 15000  # Reduce from 25000

# Clean up old worktrees
find .supervisor/.worktrees -mtime +7 -exec rm -rf {} \;
```

#### Database Corruption

**Symptoms**:
```
Error: database disk image is malformed
```

**Diagnosis**:
```bash
# Check database integrity
sqlite3 .supervisor/state.db "PRAGMA integrity_check;"
```

**Solutions**:
```bash
# Backup database
cp .supervisor/state.db .supervisor/state.db.backup

# Rebuild projections from events
sqlite3 .supervisor/state.db <<EOF
DELETE FROM workflows;
DELETE FROM steps;
DELETE FROM features;
DELETE FROM phases;
DELETE FROM components;
-- Events table is source of truth, rebuild from it
EOF

# Or restore from backup
mv .supervisor/state.db .supervisor/state.db.corrupt
mv .supervisor/state.db.backup .supervisor/state.db
```

---

## Maintenance

### Database Maintenance

#### Vacuum Database

Reclaim disk space after deletions:

```bash
# Vacuum database (reduces file size)
sqlite3 .supervisor/state.db "VACUUM;"

# Check size before/after
ls -lh .supervisor/state.db
```

**Schedule monthly**:
```bash
# Add to crontab
0 2 1 * * cd /path/to/project && sqlite3 .supervisor/state.db "VACUUM;"
```

#### Clean Old Data

Archive old workflows:

```bash
#!/bin/bash
# archive-old-workflows.sh

DB=".supervisor/state.db"
CUTOFF_DAYS=90

# Export old events
sqlite3 $DB <<EOF
.mode csv
.output old-events.csv
SELECT * FROM events WHERE timestamp < datetime('now', '-${CUTOFF_DAYS} days');
.quit
EOF

# Delete old events (be careful!)
sqlite3 $DB \
  "DELETE FROM events WHERE timestamp < datetime('now', '-${CUTOFF_DAYS} days');"

# Vacuum
sqlite3 $DB "VACUUM;"

echo "Archived events older than ${CUTOFF_DAYS} days to old-events.csv"
```

### Worktree Cleanup

Clean up stale worktrees:

```bash
#!/bin/bash
# cleanup-worktrees.sh

WORKTREE_DIR=".supervisor/.worktrees"
MAX_AGE_DAYS=7

# Find and remove old worktrees
find "$WORKTREE_DIR" -maxdepth 1 -type d -mtime +$MAX_AGE_DAYS -exec rm -rf {} \;

# Remove stale git worktree references
cd /path/to/project
git worktree prune

echo "Cleaned worktrees older than ${MAX_AGE_DAYS} days"
```

**Schedule daily**:
```bash
# Add to crontab
0 3 * * * cd /path/to/project && ./cleanup-worktrees.sh
```

### Docker Image Updates

Update Docker images regularly:

```bash
# Update base image
docker pull python:3.11-slim

# Remove old images
docker image prune -a -f --filter "until=720h"  # 30 days

# Check disk usage
docker system df
```

### Log Rotation

Configure log rotation (if using persistent logs):

```bash
# /etc/logrotate.d/supervisor
/path/to/project/.supervisor/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0640 user group
}
```

---

## Production Best Practices

### 1. Resource Limits

Configure appropriate limits:

```yaml
# .supervisor/limits.yaml
workflow_timeout: 3600      # 1 hour
component_timeout: 300      # 5 minutes

role_timeouts:
  planner: 600
  implementer: 300
  reviewer: 180

# .supervisor/config.yaml
sandbox:
  memory_limit: 2g
  cpu_limit: 1.5
  max_output_bytes: 10485760  # 10MB
```

### 2. Approval Policies

Configure approval gates for production:

```yaml
# .supervisor/approval.yaml
approval:
  auto_approve_low_risk: false  # Require approval for all in prod
  risk_threshold: low

  require_approval_for:
    - "**/*.sql"              # Database changes
    - "deploy/**"             # Deployment scripts
    - "config/production.yaml" # Production config
    - "Dockerfile"
```

### 3. Security Hardening

**Docker Security**:
```yaml
# .supervisor/config.yaml
sandbox:
  read_only_root: true
  no_new_privileges: true
  security_opt:
    - no-new-privileges
    - seccomp=default
```

**Network Security**:
- Use egress allowlist (see Deployment section)
- Restrict Docker network access
- Use VPN for AI API access (optional)

**Secrets Management**:
```bash
# Use external secrets manager
export ANTHROPIC_API_KEY=$(aws secretsmanager get-secret-value --secret-id supervisor/anthropic-key --query SecretString --output text)

# Or use encrypted files
# ansible-vault, sops, etc.
```

### 4. Backup Strategy

**Database Backup**:
```bash
#!/bin/bash
# backup-supervisor.sh

BACKUP_DIR="/backup/supervisor"
DATE=$(date +%Y%m%d-%H%M%S)

# Backup database
cp .supervisor/state.db "$BACKUP_DIR/state.db.$DATE"

# Compress old backups
find "$BACKUP_DIR" -name "state.db.*" -mtime +1 -exec gzip {} \;

# Remove backups older than 30 days
find "$BACKUP_DIR" -name "state.db.*.gz" -mtime +30 -delete

echo "Backup created: state.db.$DATE"
```

**Schedule backups**:
```bash
# Hourly backups
0 * * * * cd /path/to/project && ./backup-supervisor.sh

# Daily backups to S3
0 2 * * * cd /path/to/project && aws s3 sync /backup/supervisor s3://your-bucket/supervisor-backups/
```

### 5. Monitoring and Alerting

**Health Check Endpoint** (if running as service):
```python
# health.py
from supervisor.core.state import Database

def health_check():
    try:
        db = Database()
        # Check if database is accessible
        db.get_events("health-check")
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

**Prometheus Metrics** (optional):
```python
# Export metrics for Prometheus
# supervisor/metrics/prometheus.py

from prometheus_client import Counter, Histogram, Gauge

executions_total = Counter('supervisor_executions_total', 'Total executions', ['role', 'status'])
execution_duration = Histogram('supervisor_execution_duration_seconds', 'Execution duration', ['role'])
workflows_active = Gauge('supervisor_workflows_active', 'Active workflows')
```

### 6. CI/CD Integration

**GitHub Actions**:
```yaml
# .github/workflows/supervisor-review.yml
name: AI Code Review

on: [pull_request]

jobs:
  ai-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Supervisor
        run: |
          pip install -e .
          supervisor init

      - name: Run AI Review
        run: supervisor run reviewer "Review PR changes"
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}

      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: review-results
          path: .supervisor/
```

---

## Scaling

### Horizontal Scaling

**Multi-Machine Setup**:

1. **Shared Database** (switch from SQLite to PostgreSQL):
```bash
# Install PostgreSQL adapter
pip install psycopg2-binary

# Configure connection
export SUPERVISOR_DB_URL="postgresql://user:pass@db-host:5432/supervisor"
```

2. **Distributed Worktrees** (shared filesystem):
```bash
# Mount NFS/GlusterFS for worktrees
mount -t nfs nfs-server:/supervisor-worktrees /path/to/.supervisor/.worktrees
```

3. **Load Balancing**:
```bash
# HAProxy configuration
backend supervisor_workers
    balance roundrobin
    server worker1 worker1:8000 check
    server worker2 worker2:8000 check
    server worker3 worker3:8000 check
```

### Vertical Scaling

**Optimize Resource Usage**:

```yaml
# .supervisor/config.yaml
workflow:
  parallel_execution: true
  max_parallel_components: 4  # Limit concurrent components

sandbox:
  memory_limit: 4g  # Increase memory
  cpu_limit: 2.0    # Increase CPU
```

**Use Faster Storage**:
- SSD for database and worktrees
- tmpfs for temporary files

---

## Disaster Recovery

### Backup and Restore

**Full Backup**:
```bash
#!/bin/bash
# Full backup of Supervisor state

BACKUP_DIR="/backup/supervisor/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup database
cp .supervisor/state.db "$BACKUP_DIR/"

# Backup configuration
cp -r .supervisor/*.yaml "$BACKUP_DIR/"

# Backup custom roles and gates
cp -r .supervisor/roles "$BACKUP_DIR/"

# Create tarball
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Full backup created: $BACKUP_DIR.tar.gz"
```

**Restore**:
```bash
#!/bin/bash
# Restore from backup

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    exit 1
fi

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/

# Restore database
cp /tmp/*/state.db .supervisor/

# Restore configuration
cp /tmp/*/*.yaml .supervisor/

# Restore custom roles
cp -r /tmp/*/roles .supervisor/

echo "Restored from: $BACKUP_FILE"
```

### Recovery Procedures

**Database Corruption**:
1. Stop all Supervisor processes
2. Restore from latest backup
3. Verify integrity: `sqlite3 .supervisor/state.db "PRAGMA integrity_check;"`
4. Rebuild projections if needed

**Lost Worktree**:
1. Check git worktree list: `git worktree list`
2. Remove stale references: `git worktree prune`
3. Workflow will recreate worktree on retry

**Configuration Loss**:
1. Restore from backup
2. Or reinitialize: `supervisor init`
3. Reconfigure custom settings

---

## Performance Tuning

### Optimization Checklist

- [ ] Enable gate caching (`.supervisor/gates.yaml`: `cache: true`)
- [ ] Use appropriate model for task (Haiku for simple, Opus for complex)
- [ ] Configure parallel execution (`workflow.parallel_execution: true`)
- [ ] Reduce context budget for faster roles (`context.token_budget: 15000`)
- [ ] Clean up old worktrees regularly
- [ ] Vacuum database monthly
- [ ] Use SSD for `.supervisor/` directory
- [ ] Increase Docker resources if needed
- [ ] Monitor and optimize slow gates

### Benchmarking

```bash
# Benchmark workflow execution
time supervisor workflow feat-test-123

# Benchmark specific role
time supervisor run implementer "Test task" -t test.py

# Profile with metrics
supervisor metrics --days 1 > before.txt
# Run workflows
supervisor metrics --days 1 > after.txt
diff before.txt after.txt
```

---

## Security Checklist

- [ ] Docker is required and running
- [ ] Network egress is configured and verified
- [ ] API keys are stored securely (not in git)
- [ ] Approval policies are configured for production
- [ ] Database backups are automated
- [ ] Log rotation is configured
- [ ] Resource limits are set
- [ ] Security gates are enabled (bandit, etc.)
- [ ] Docker images are regularly updated
- [ ] Access control is implemented (file permissions)

---

## Support and Escalation

For issues not covered in this guide:

1. Check logs: `.supervisor/logs/` or `docker logs`
2. Query database: `sqlite3 .supervisor/state.db`
3. Review metrics: `supervisor metrics`
4. Check GitHub issues: https://github.com/your-org/supervisor/issues
5. Contact support: support@your-org.com

---

## See Also

- [Getting Started](GETTING_STARTED.md) - Quickstart guide
- [CLI Reference](CLI_REFERENCE.md) - Command documentation
- [Architecture](ARCHITECTURE.md) - System design
- [Contributing](../CONTRIBUTING.md) - Development guide
