# PersistentVolumeClaim stuck in Pending

## Symptoms
The pod is `Pending` with the event `pod has unbound immediate PersistentVolumeClaims`, and `kubectl get pvc` shows the claim as `Pending`. The PVC's events say `no persistent volumes available for this claim and no storage class is set`, or `waiting for first consumer to be created before binding`.

## Cause
No volume can be provisioned or bound: the cluster has no default StorageClass, the claim names a `storageClassName` that does not exist, or the CSI provisioner is failing. With `volumeBindingMode: WaitForFirstConsumer`, Pending is normal until a pod that uses the claim is scheduled.

## Fix
List the classes with `kubectl get storageclass` and check which one is marked default. Set the correct `storageClassName` in the claim. If the class exists, read the logs of the CSI provisioner pods for the real error.
