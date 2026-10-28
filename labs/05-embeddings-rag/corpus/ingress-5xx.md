# Ingress returns 502 or 503

## Symptoms
Requests through the NGINX ingress controller fail with `503 Service Temporarily Unavailable` or `502 Bad Gateway`, while the pods themselves look healthy.

## Cause
A 503 usually means the Service has no ready endpoints: its selector does not match the pod labels, or all pods are unready. A 502 means the controller reached a pod but the connection failed: the Service's `targetPort` is wrong, or the application listens only on `127.0.0.1` instead of all interfaces.

## Fix
Check `kubectl get endpointslices -l kubernetes.io/service-name=<service>` and compare the Service selector with the pod labels. Verify that `targetPort` matches the container port, and read the ingress controller's logs for the upstream error.
