param(
    [string[]]$Models = @('qwen3.5:4b'),
    [string]$OutDir = 'results/raw'   # gitignored
)

# Step 6. Grounding: does putting the facts in the prompt fix what the model got wrong?
# Questions the models answered wrongly in Steps 1 and 4, under six conditions:
#   closed            question only (Step 1 again)
#   grounded          correct reference + question
#   grounded-strict   + "answer only from the reference, else reply NOT IN REFERENCE"
#   irrelevant-strict an unrelated reference + the strict rule: does it abstain?
#   poisoned          a plausible but WRONG reference: does it follow the text or its knowledge?
#   poisoned-strict   wrong reference + the strict rule
# Temperature 0, thinking off: one deterministic answer per cell.

$api = 'http://127.0.0.1:11434'

# Question text and Checks/Wrong patterns come from quality.ps1 (one source of truth), plus api-port from Step 4b
$ast = [System.Management.Automation.Language.Parser]::ParseFile((Join-Path $PSScriptRoot 'quality.ps1'), [ref]$null, [ref]$null)
$assign = $ast.Find({ $args[0] -is [System.Management.Automation.Language.AssignmentStatementAst] -and $args[0].Left.Extent.Text -eq '$questions' }, $true)
$byId = @{}; foreach ($q in (Invoke-Expression $assign.Right.Extent.Text)) { $byId[$q.Id] = $q }
$byId['api-port'] = @{ Id = 'api-port'
    Q = 'What is the default secure port of the Kubernetes API server? Answer briefly.'
    Checks = @('\b6443\b'); Wrong = @('(?<!\d)443\b') }   # note: also hits a correct answer that mentions 443 for managed clusters

# References. Correct ones paraphrase the official docs (sources in the Lab 03 README).
# Poisoned ones are plausible and wrong; Poison = patterns that show the model repeated the poison.
$items = @(
    @{ Id = 'helm-state'
       Ref    = 'Helm 3 stores release information as Kubernetes Secrets in the namespace of the release. Secrets are the default storage backend; the HELM_DRIVER environment variable can select configmap, secret or sql (beta) instead. (Helm 2 stored release information in ConfigMaps in the kube-system namespace, through its server-side component Tiller.)'
       Poison = 'Helm 3 stores release information as ConfigMaps in the kube-system namespace, managed by the helm-controller.'
       PoisonPat = @('ConfigMap', 'kube-system', 'helm-controller') }
    @{ Id = 'probe-defaults'
       Ref    = 'Probe fields: periodSeconds: How often (in seconds) to perform the probe. Default to 10 seconds. Minimum value is 1. failureThreshold: Minimum consecutive failures for the probe to be considered failed after having succeeded. Defaults to 3. Minimum value is 1.'
       Poison = 'Probe fields: periodSeconds: how often (in seconds) to perform the probe. Defaults to 30 seconds. failureThreshold: the number of consecutive failures before the check is considered failed. Defaults to 5.'
       PoisonPat = @('periodSeconds\D{0,40}\b30\b', 'failureThreshold\D{0,40}\b5\b') }
    @{ Id = 'api-port'
       Ref    = 'kube-apiserver flag --secure-port int, default 6443: the port on which to serve HTTPS with authentication and authorization. It cannot be switched off with 0.'
       Poison = 'kube-apiserver flag --secure-port int, default 8443: the port on which to serve HTTPS with authentication and authorization.'
       PoisonPat = @('\b8443\b') }
    @{ Id = 'fake-flag'
       Ref    = 'Commonly used kubectl apply flags: -f/--filename, -k/--kustomize, -R/--recursive, --dry-run, --server-side, --force-conflicts, --field-manager, --prune, -l/--selector, --force, --grace-period, --timeout, --wait, -o/--output, --validate. kubectl apply has no self-healing option; continuous reconciliation is done by controllers or GitOps tools such as Argo CD.'
       Poison = 'kubectl apply flags: -f/--filename, --dry-run, --server-side, --prune, --auto-heal (keeps watching the applied objects and re-applies the manifest whenever the live state drifts).'
       PoisonPat = @('re-?appl', 'drift', 'keeps watching') }
)
# Unrelated reference for irrelevant-strict: real content, wrong topic
$irrelevant = 'HorizontalPodAutoscaler: the controller checks metrics every 15 seconds by default (--horizontal-pod-autoscaler-sync-period) and scales the target between minReplicas and maxReplicas. The default downscale stabilization window is 300 seconds.'
$strict = 'Answer only from the reference. If the reference does not contain the answer, reply exactly: NOT IN REFERENCE'
$conditions = @(
    @{ Name = 'closed';            Ref = { param($i) $null };         Strict = $false }
    @{ Name = 'grounded';          Ref = { param($i) $i.Ref };        Strict = $false }
    @{ Name = 'grounded-strict';   Ref = { param($i) $i.Ref };        Strict = $true }
    @{ Name = 'irrelevant-strict'; Ref = { param($i) $irrelevant };   Strict = $true }
    @{ Name = 'poisoned';          Ref = { param($i) $i.Poison };     Strict = $false }
    @{ Name = 'poisoned-strict';   Ref = { param($i) $i.Poison };     Strict = $true }
)

New-Item -ItemType Directory -Force $OutDir | Out-Null
$report = Join-Path $OutDir "grounding-$(Get-Date -Format 'yyyyMMdd-HHmmss').md"
$total = $Models.Count * $conditions.Count * $items.Count
$n = 0

$rows = foreach ($model in $Models) {
    Add-Content $report "# $model`n"
    foreach ($c in $conditions) {
        foreach ($i in $items) {
            Write-Progress -Activity "grounding" -Status "Request $($n + 1) of ${total}: $model, $($c.Name), $($i.Id)" -PercentComplete ([int](100 * $n / $total))
            $n++
            $q = $byId[$i.Id]
            $ref = & $c.Ref $i
            # The reference is delimited and labelled as data; the question comes last (Step 5: the end survives)
            $user = if ($ref) { "Reference:`n<<<`n$ref`n>>>`n`nQuestion: $($q.Q)" } else { $q.Q }
            $msgs = @()
            if ($c.Strict) { $msgs += @{ role = 'system'; content = $strict } }
            $msgs += @{ role = 'user'; content = $user }
            $body = @{ model = $model; messages = $msgs; stream = $false; think = $false
                       options = @{ temperature = 0; seed = 42; num_predict = 300 } } | ConvertTo-Json -Depth 5
            $a = (Invoke-RestMethod "$api/api/chat" -Method Post -Body $body -ContentType 'application/json').message.content.Trim()

            $hits    = @($q.Checks | Where-Object { $a -match $_ }).Count
            $wrong   = @($q.Wrong  | Where-Object { $a -match $_ }).Count
            # Poison only counts where poison was given: the correct Helm reference mentions Helm 2's ConfigMaps on purpose
            $poison  = if ($c.Name -like 'poisoned*') { @($i.PoisonPat | Where-Object { $a -match $_ }).Count } else { 0 }
            $abstain = $a -match 'NOT IN REFERENCE'
            $complete = $hits -eq $q.Checks.Count
            $verdict = if ($abstain) { 'abstain' } elseif ($poison) { 'POISON' }
                       elseif ($complete -and -not $wrong) { 'correct' }
                       elseif ($wrong -and -not $complete) { 'wrong' }
                       else { 'check' }   # partial, or right answer plus a wrong-claim hit (e.g. Helm 2 contrast): read it
            Add-Content $report "## $($c.Name) / $($i.Id) ($verdict; score $hits/$($q.Checks.Count), wrong $wrong, poison $poison)`n`n$a`n"
            [pscustomobject]@{
                Model     = $model
                Condition = $c.Name
                Question  = $i.Id
                Verdict   = $verdict   # 'check' = no rule matched: read the answer
                Score     = "$hits/$($q.Checks.Count)"
                Wrong     = $wrong
                Poison    = $poison
                Answer    = ($a -replace '\s+', ' ').Substring(0, [math]::Min(60, ($a -replace '\s+', ' ').Length))
            }
        }
    }
}
Write-Progress -Activity "grounding" -Completed
$rows | Format-Table -AutoSize

# Compact view: one row per condition, one column per question
"Verdicts:"
$rows | Group-Object Model, Condition | ForEach-Object {
    $o = [ordered]@{ Model = $_.Group[0].Model; Condition = $_.Group[0].Condition }
    foreach ($r in $_.Group) { $o[$r.Question] = $r.Verdict }
    [pscustomobject]$o
} | Format-Table -AutoSize
"Answers saved to $report"
