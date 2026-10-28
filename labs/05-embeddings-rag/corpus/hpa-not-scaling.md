# HorizontalPodAutoscaler does not scale

## Symptoms
`kubectl get hpa` shows `TARGETS <unknown>/80%` and the replica count never changes, even under load. Events say `failed to get cpu utilization: unable to get metrics for resource cpu`. `kubectl top pods` fails with `Metrics API not available`.

## Cause
The HPA reads CPU and memory from the metrics API, which is provided by metrics-server. It cannot scale when metrics-server is missing or failing, or when the pods have no CPU request: utilization is measured relative to the request. It also stops at `maxReplicas`.

## Fix
Install or repair metrics-server and check `kubectl top pods` again. Set `resources.requests.cpu` on every container of the target workload. Check `maxReplicas` and the HPA's conditions with `kubectl describe hpa`.
