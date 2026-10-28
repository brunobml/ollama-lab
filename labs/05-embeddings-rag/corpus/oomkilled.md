# Container OOMKilled

## Symptoms
`kubectl describe pod` shows `Last State: Terminated`, `Reason: OOMKilled`, `Exit Code: 137`. The container restarts, and its memory graph rises until it hits the same ceiling every time (a saw-tooth pattern).

## Cause
The container used more memory than its `resources.limits.memory`. The limit is enforced by the cgroup, and the kernel's OOM killer terminates the process with SIGKILL; exit code 137 is 128 + 9 (signal 9). The node itself may have plenty of free memory: this is about the container's own limit.

## Fix
Compare actual usage (`kubectl top pod`) with the limit. Raise `resources.limits.memory` if the workload legitimately needs more, or fix the leak. For the JVM, size the heap from the container limit (for example `-XX:MaxRAMPercentage=75`) instead of a fixed `-Xmx` equal to the limit, because the JVM needs memory beyond the heap.
