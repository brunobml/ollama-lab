# Pod in CrashLoopBackOff

## Symptoms
`kubectl get pods` shows STATUS `CrashLoopBackOff` and the RESTARTS counter keeps increasing. The kubelet waits longer before each restart: the back-off delay doubles (10s, 20s, 40s, ...) up to five minutes. Between restarts the pod may briefly show `Running` or `Error`.

## Cause
The container's main process exits, and the pod's `restartPolicy: Always` makes the kubelet start it again. Common reasons: a configuration error at startup, a missing environment variable or Secret, a wrong `command` or `args`, or a dependency the application refuses to start without.

## Fix
Read the logs of the previous, crashed instance with `kubectl logs <pod> --previous`; the current instance may not have logged anything yet. `kubectl describe pod <pod>` shows `Last State: Terminated` with the `Reason` and `Exit Code`. Fix the configuration, then let the next restart pick it up or roll out a new version.
