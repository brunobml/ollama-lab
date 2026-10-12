param(
    [Parameter(Mandatory)][string[]]$Models,
    [int]$NumPredict = 400,
    [string]$OutDir = 'results/raw'   # gitignored
)

# Fixed question set. Checks are case-insensitive regexes: a crude, repeatable signal,
# not a real benchmark. Always read the saved answers too.
$questions = @(
    @{ Id = 'probe-defaults'
       Q  = 'In Kubernetes, what are the default periodSeconds and failureThreshold of a readinessProbe? Answer briefly.'
       Checks = @('periodSeconds\D{0,40}\b10\b', 'failureThreshold\D{0,40}\b3\b') }
    @{ Id = 'readiness-vs-liveness'
       Q  = 'What happens to a Pod when its readinessProbe fails, and how does that differ from a failing livenessProbe? Answer briefly.'
       Checks = @('endpoint', "(not|n't|never|without) (be )?restart", 'liveness[^.]*restart') }
    @{ Id = 'helm-state'
       Q  = 'Where does Helm 3 store release information by default? Answer briefly.'
       Checks = @('secret', 'namespace') }
    @{ Id = 'argocd-selfheal'
       Q  = 'In Argo CD, what does syncPolicy.automated.selfHeal do? Answer briefly.'
       Checks = @('(drift|manual|out.of.sync|live (state|cluster))', '(git|desired state)') }
    @{ Id = 'sched-math'
       Q  = 'A node has 4 CPU allocatable. Running pods request 1500m, 1200m and 800m CPU. Can a new pod requesting 600m CPU be scheduled on it? Show the arithmetic briefly.'
       Checks = @('(3500m|3\.5)', '(500m|0\.5)', '\b(no|cannot|can.t|not be scheduled|won.t)\b') }
    @{ Id = 'fake-flag'
       Q  = 'What does the kubectl apply flag --auto-heal do? Answer briefly.'
       Checks = @("(does not exist|doesn.t exist|no such|not a (valid|real|standard)|not aware|unknown|isn.t a|is not a)") }
)

New-Item -ItemType Directory -Force $OutDir | Out-Null
$stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$report = Join-Path $OutDir "quality-$stamp.md"
$first  = @{}   # answers of the first model, to spot wording changes

$results = foreach ($model in $Models) {
    Add-Content $report "# $model`n"
    foreach ($q in $questions) {
        $body = @{
            model   = $model
            prompt  = $q.Q
            stream  = $false
            think   = $false
            options = @{ num_predict = $NumPredict; num_ctx = 4096; temperature = 0; seed = 42 }
        } | ConvertTo-Json -Depth 4
        $r = Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json'

        $hits = @($q.Checks | Where-Object { $r.response -match $_ }).Count
        if (-not $first.ContainsKey($q.Id)) { $first[$q.Id] = $r.response }
        Add-Content $report "## $($q.Id) ($hits/$($q.Checks.Count))`n`n$($r.response)`n"

        [pscustomobject]@{
            Model    = $model
            Question = $q.Id
            Score    = "$hits/$($q.Checks.Count)"
            GenTok   = $r.eval_count
            GenTokS  = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1)
            SameText = $r.response -eq $first[$q.Id]
        }
    }
}
$results | Format-Table -AutoSize
"Answers saved to $report"
