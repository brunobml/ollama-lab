# TLS certificate expired

## Symptoms
Clients fail with `x509: certificate has expired or is not yet valid`, and browsers show `NET::ERR_CERT_DATE_INVALID`. The Ingress keeps serving the old certificate.

## Cause
The certificate was not renewed in time. With cert-manager, renewal fails when the ACME challenge cannot complete (for example the HTTP-01 challenge path is not reachable or the DNS-01 credentials are wrong) or when the Issuer is misconfigured. Certificates stored manually in a Secret are never renewed at all.

## Fix
Inspect the chain of resources: `kubectl get certificate,certificaterequest,order,challenge -n <namespace>` and `kubectl describe` the one that is not ready. Check the expiry date of the served certificate with `openssl x509 -noout -enddate`. After fixing the cause, trigger a renewal with `cmctl renew <certificate>`.
