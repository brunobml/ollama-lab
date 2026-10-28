# Pod Pending: no node fits

## Symptoms
The pod stays `Pending` and is never assigned to a node. `kubectl describe pod` shows a `FailedScheduling` event such as `0/3 nodes are available: 3 Insufficient cpu.` or `node(s) had untolerated taint`.

## Cause
The scheduler found no node that satisfies the pod: its CPU or memory requests exceed the free allocatable capacity of every node, or nodeSelector, affinity rules or taints exclude the nodes that would have room.

## Fix
Compare the pod's requests with `kubectl describe node` (section `Allocated resources`). Lower the requests if they are oversized, add nodes or let the cluster autoscaler add them, or add the missing toleration or fix the node selector.
