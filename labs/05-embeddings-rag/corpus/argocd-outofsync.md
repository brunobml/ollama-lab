# Argo CD application keeps returning to OutOfSync

## Symptoms
Right after a successful sync the application shows `OutOfSync` again. The diff shows fields that nobody changed in Git, such as `spec.replicas` or defaulted fields; with self-heal enabled, the value flips back and forth on every sync.

## Cause
Something other than Argo CD modifies the live object: a HorizontalPodAutoscaler changes the replica count, a mutating webhook injects fields, or an operator updates its own resources. The live state then never matches the desired state in Git.

## Fix
Remove fields that another controller owns from the manifests (for example drop `replicas` when an HPA manages the Deployment), or tell Argo CD to ignore them with `ignoreDifferences` (for example a `jsonPointers` entry for `/spec/replicas`) in the Application spec. Server-side diff also avoids many false differences from defaulted fields.
