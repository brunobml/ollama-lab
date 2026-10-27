param(
    [Parameter(Mandatory)][string[]]$Models,
    [int]$NumPredict = 400,
    [int]$NumCtx = 4096,
    [switch]$Think,                    # thinking mode on (needs a larger NumPredict and NumCtx)
    [string]$OutDir = 'results/raw'   # gitignored
)

# Fixed question set. Checks are case-insensitive regexes: a crude, repeatable signal,
# not a real benchmark. Always read the saved answers too.
#   Checks: things a correct answer should contain (Score = hits / checks)
#   Wrong:  known wrong claims; any hit is counted in the Wrong column
$questions = @(
    @{ Id = 'probe-defaults'
       Q  = 'In Kubernetes, what are the default periodSeconds and failureThreshold of a readinessProbe? Answer briefly.'
       Checks = @('periodSeconds\D{0,40}\b10\b', 'failureThreshold\D{0,40}\b3\b')
       Wrong  = @('(does|do)\W+not\W+(define|have|set)', "doesn.t\W+(define|have|set)", 'no default', 'has\W+no\W+`?(periodSeconds|failureThreshold|default)', '\bnull\b') }   # \W+ skips Markdown like **not**
    @{ Id = 'readiness-vs-liveness'
       Q  = 'What happens to a Pod when its readinessProbe fails, and how does that differ from a failing livenessProbe? Answer briefly.'
       Checks = @('endpoint', "((not|n't|never|without) (be )?restart|(keeps?|continues|remains|stays)[^.]{0,30}running)", 'liveness[^.]*restart')
       Wrong  = @('CoreDNS', 'scheduler stops') }
    @{ Id = 'helm-state'
       Q  = 'Where does Helm 3 store release information by default? Answer briefly.'
       Checks = @('secret', 'namespace')
       Wrong  = @('kube-system', 'ConfigMap', '/var/lib/helm', '[/\\]\.helm\b', 'kube-state-metrics', 'leases\.helm') }
    @{ Id = 'argocd-selfheal'
       Q  = 'In Argo CD, what does syncPolicy.automated.selfHeal do? Answer briefly.'
       Checks = @('(drift|manual|out.of.sync|live (state|cluster))', '(git|desired state)')
       Wrong  = @('changes[^\n]{0,80}back to (the |your )?git') }   # wrong direction: self-heal reverts the cluster to Git
    @{ Id = 'sched-math'
       Q  = 'A node has 4 CPU allocatable. Running pods request 1500m, 1200m and 800m CPU. Can a new pod requesting 600m CPU be scheduled on it? Show the arithmetic briefly.'
       # \s*(\\text\{\s*)? tolerates LaTeX answers like 3500\text{m}
       Checks = @('(3500\s*(\\text\{\s*)?m|3\.5)', '(\b500\s*(\\text\{\s*)?m|0\.5)', '\b(no|cannot|can.t|not be scheduled|won.t)\b')
       Wrong  = @('^\W*yes\b') }       # verdict "Yes" in the first word contradicts the math
    @{ Id = 'fake-flag'
       Q  = 'What does the kubectl apply flag --auto-heal do? Answer briefly.'
       Checks = @("(does not exist|doesn.t exist|there is no|no such|not a (valid|real|standard)|not aware|unknown|isn.t a|is not a)")
       Wrong  = @('^(the )?`?--auto-heal`? flag (tells|makes|enables|causes|instructs)') }
)

New-Item -ItemType Directory -Force $OutDir | Out-Null
$stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$report = Join-Path $OutDir "quality-$stamp.md"
$first  = @{}   # answers of the first model, to spot wording changes

$total = $Models.Count * $questions.Count
$n = 0
$results = foreach ($model in $Models) {
    Add-Content $report "# $model (think=$([bool]$Think))`n"
    foreach ($q in $questions) {
        Write-Progress -Activity "quality: think=$([bool]$Think)" -Status "Request $($n + 1) of ${total}: $model, $($q.Id)" -PercentComplete ([int](100 * $n / $total))
        $n++
        $body = @{
            model   = $model
            prompt  = $q.Q
            stream  = $false
            think   = [bool]$Think
            options = @{ num_predict = $NumPredict; num_ctx = $NumCtx; temperature = 0; seed = 42 }
        } | ConvertTo-Json -Depth 4
        $r = Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json'

        $hits  = @($q.Checks | Where-Object { $r.response -match $_ }).Count
        $wrong = @($q.Wrong  | Where-Object { $r.response -match $_ }).Count
        if (-not $first.ContainsKey($q.Id)) { $first[$q.Id] = $r.response }
        $thinkText = if ($r.thinking) { "<details><summary>thinking</summary>`n`n$($r.thinking)`n`n</details>`n`n" } else { '' }
        Add-Content $report "## $($q.Id) ($hits/$($q.Checks.Count), wrong $wrong, $($r.done_reason))`n`n$thinkText$($r.response)`n"

        [pscustomobject]@{
            Model       = $model
            Question    = $q.Id
            Score       = "$hits/$($q.Checks.Count)"
            Wrong       = $wrong
            GenTok      = $r.eval_count                    # includes thinking tokens
            ThinkChars  = "$($r.thinking)".Length
            AnswerChars = "$($r.response)".Length
            Seconds     = [math]::Round($r.total_duration / 1e9, 1)
            Done        = $r.done_reason                   # 'length' = cut off by num_predict
            SameText    = $r.response -eq $first[$q.Id]
        }
    }
}
Write-Progress -Activity "quality: think=$([bool]$Think)" -Completed
$results | Format-Table -AutoSize
"Answers saved to $report"
