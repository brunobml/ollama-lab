# Readiness probe failing

## Symptoms
The pod is `Running` but READY shows `0/1`, and it receives no traffic. Events show `Readiness probe failed: HTTP probe failed with statuscode: 503`. The container is not restarted: the RESTARTS counter stays the same.

## Cause
A failing readiness probe removes the pod from the endpoints of every Service that selects it; it never restarts the container. The application is not ready yet, a dependency it checks is down, or the probe uses the wrong path or port.

## Fix
Check whether the pod is listed with `kubectl get endpointslices -l kubernetes.io/service-name=<service>`. Call the probe's path from inside the pod to see the real response. Correct the path or port, or give a slow application more time with `initialDelaySeconds` or a startup probe.
