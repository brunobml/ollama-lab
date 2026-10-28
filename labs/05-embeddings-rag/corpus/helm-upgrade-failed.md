# Helm upgrade blocked: another operation is in progress

## Symptoms
`helm upgrade` fails with `Error: UPGRADE FAILED: another operation (install/upgrade/rollback) is in progress`, even though no Helm command is running. `helm history` shows the last revision in status `pending-upgrade` or `pending-install`.

## Cause
A previous Helm process was interrupted before it could record the result, for example a CI job that was cancelled or timed out in the middle of an upgrade. The release stays locked in its pending state.

## Fix
Confirm that no deployment job is really still running. Then roll back to the last revision with status `deployed` (`helm rollback <release> <revision>`), which clears the pending state, and run the upgrade again.
