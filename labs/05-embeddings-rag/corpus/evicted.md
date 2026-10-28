# Pods evicted under node memory pressure

## Symptoms
Pods end in phase `Failed` with reason `Evicted` and the message `The node was low on resource: memory.` Often many pods from the same node are affected at once, and the node reports the condition `MemoryPressure`.

## Cause
The kubelet performs node-pressure eviction when the node's available memory falls below an eviction threshold (by default a hard threshold of `memory.available<100Mi` on Linux). It evicts pods to protect the node, starting with pods whose usage exceeds their requests: BestEffort first, then Burstable.

## Fix
Set memory requests close to real usage so the scheduler does not overpack nodes, and give critical workloads Guaranteed QoS (requests equal to limits). Add capacity if nodes are consistently full. Evicted pods are not deleted automatically; clean them up with `kubectl delete pods --field-selector=status.phase=Failed`.
