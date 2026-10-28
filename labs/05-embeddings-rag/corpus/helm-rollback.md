# Roll back a Helm release

## Symptoms
A `helm upgrade` was applied successfully, but the new version of the application misbehaves and you need the previous version back quickly.

## Cause
Helm stores every installed version of a release as a numbered revision, so an earlier configuration can be reapplied without re-running the old pipeline.

## Fix
List the revisions with `helm history <release> -n <namespace>`. Run `helm rollback <release> <revision>`; if the revision is omitted, Helm rolls back to the previous release. The rollback itself creates a new revision. Note that a rollback does not revert CRDs or data in persistent volumes.
