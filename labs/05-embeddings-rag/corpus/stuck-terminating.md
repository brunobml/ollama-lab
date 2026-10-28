# PVC or namespace stuck in Terminating

## Symptoms
A PersistentVolumeClaim stays in `Terminating` after deletion, or deleting a namespace never finishes and the namespace remains `Terminating` indefinitely.

## Cause
Finalizers block the deletion until their controller has done its cleanup. The `kubernetes.io/pvc-protection` finalizer keeps a PVC while any pod still uses it. A namespace waits until every resource inside it is gone, so a resource whose finalizer belongs to a controller that was uninstalled blocks it forever.

## Fix
For a PVC, find and delete the pods that still mount it. For a namespace, list what is left (`kubectl api-resources --verbs=list --namespaced -o name` with `kubectl get -n <namespace>`) and check its finalizers. Remove a finalizer by hand only after confirming that its controller is really gone, because the cleanup it guards (for example deleting cloud volumes) will then never happen.
