# ImagePullBackOff and ErrImagePull

## Symptoms
The pod shows `ErrImagePull`, then `ImagePullBackOff` while the kubelet retries with a growing delay. Events contain messages such as `manifest unknown`, `not found`, `pull access denied`, `unauthorized: authentication required`, or `toomanyrequests` from Docker Hub.

## Cause
The kubelet cannot pull the image: the name or tag is wrong or was never pushed, the registry is private and no credentials were provided, or the registry rate-limits anonymous pulls.

## Fix
Check that the exact tag exists in the registry. For a private registry, create a Secret of type `kubernetes.io/dockerconfigjson` (`kubectl create secret docker-registry`) and reference it in `imagePullSecrets` of the pod or its ServiceAccount. For rate limits, authenticate the pulls or mirror the image into your own registry.
