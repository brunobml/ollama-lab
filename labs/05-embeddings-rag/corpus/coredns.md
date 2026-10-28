# DNS resolution failing inside pods

## Symptoms
Applications log errors such as `dial tcp: lookup orders-db on 10.96.0.10:53: no such host` or `i/o timeout` when connecting to other services by name. `nslookup` from inside a pod fails or is very slow.

## Cause
Name resolution goes through the cluster DNS service (CoreDNS). It fails when the CoreDNS pods are down or crash-looping, when a NetworkPolicy blocks egress to port 53 (UDP and TCP) in `kube-system`, or when the name is wrong: a Service in another namespace needs `<service>.<namespace>`. The default `ndots:5` setting can also make lookups of external names slow.

## Fix
Check the DNS pods with `kubectl -n kube-system get pods -l k8s-app=kube-dns` and their logs. Test from a debug pod with `nslookup <service>.<namespace>.svc.cluster.local`. Allow DNS egress in your NetworkPolicies, and use fully qualified names where it matters.
