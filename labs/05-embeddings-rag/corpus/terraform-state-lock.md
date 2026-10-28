# Terraform state lock

## Symptoms
`terraform plan` or `apply` fails with `Error acquiring the state lock` and shows a lock ID, the user and the operation that holds it. With an S3 backend and DynamoDB locking the error includes `ConditionalCheckFailedException`.

## Cause
Terraform locks the state so that two runs cannot change it at the same time. Either another run really is in progress, or an earlier run crashed or was killed and left its lock behind.

## Fix
First make sure no other plan or apply is running, for example in CI. If the lock is stale, remove it with `terraform force-unlock <LOCK_ID>`. Never force-unlock while another apply is active: two concurrent writes can corrupt the state.
