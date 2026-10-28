# Node NotReady

## Symptoms
`kubectl get nodes` shows a node as `NotReady`. Pods on it switch to `Unknown` or stay `Terminating`, and after about five minutes they are evicted and recreated elsewhere (the `node.kubernetes.io/unreachable` and `not-ready` taints have a default toleration of 300 seconds).

## Cause
The control plane stopped receiving status updates from the kubelet: the kubelet or the container runtime has stopped, the node ran out of disk (`DiskPressure`), the network between node and API server is broken, or the kubelet's client certificate expired.

## Fix
On the node, check `systemctl status kubelet` and `journalctl -u kubelet`, the container runtime (`systemctl status containerd`), and free disk space with `df -h`. `kubectl describe node` lists the conditions and the last heartbeat time.
