# Liveness probe restarting containers

## Symptoms
Events show `Liveness probe failed` followed by `Container <name> failed liveness probe, will be restarted`. The container restarts regularly, often during startup or under heavy load. The previous instance ended with exit code 143 (it stopped on SIGTERM) or 137 (it was killed after the grace period), and the reason is not OOMKilled.

## Cause
When the liveness probe fails `failureThreshold` times in a row, the kubelet kills the container and starts it again. Probes that are too strict for a slow-starting application, a `timeoutSeconds` that is too short under load, or a liveness check that also tests external dependencies cause restart loops of healthy processes.

## Fix
Add a startup probe so liveness checks only begin after the application has started. Raise `timeoutSeconds` and `failureThreshold`. Make the liveness endpoint check only that the process itself is healthy, and leave dependency checks to the readiness probe.
