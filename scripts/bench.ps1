param(
    [Parameter(Mandatory)][string]$Model,
    [int]$Runs = 2,
    [int]$NumPredict = 256,
    [int[]]$NumCtx = @(4096),
    [int]$LogLines = 0,
    [int[]]$NumGpu = @(-1),   # layers on GPU; -1 = let Ollama decide (all that fit)
    [string]$Prompt = 'Explain step by step how the Kubernetes scheduler places a pod on a node.'
)

# Synthetic pod logs: measured ≈42 tokens per line (Lab 02 step 3)
$logs = if ($LogLines) {
    (1..$LogLines | ForEach-Object {
        "2026-10-10T12:{0:00}:{1:00}Z pod/api-{2} ns=prod level=INFO msg=`"request served`" latency_ms={3}" -f ($_ % 60), ($_ % 60), ($_ % 7), (Get-Random -Maximum 900)
    }) -join "`n"
} else { '' }

$total = $NumCtx.Count * $NumGpu.Count * $Runs
$n = 0
$results = foreach ($ctx in $NumCtx) {
  foreach ($gpu in $NumGpu) {
    1..$Runs | ForEach-Object {
        $gpuText = if ($gpu -ge 0) { $gpu } else { 'auto' }
        Write-Progress -Activity "bench: $Model" -Status "Request $($n + 1) of ${total}: num_ctx $ctx, num_gpu $gpuText, run $_ of $Runs" -PercentComplete ([int](100 * $n / $total))
        $n++
        $options = @{ num_predict = $NumPredict; num_ctx = $ctx; temperature = 0; seed = 42 }
        if ($gpu -ge 0) { $options.num_gpu = $gpu }   # undocumented option; maps to llama-server -ngl

        $body = @{
            model   = $Model
            prompt  = "[run $_ $(Get-Random)]`n$logs`n$Prompt"
            stream  = $false
            think   = $false
            options = $options
        } | ConvertTo-Json -Depth 4

        $r  = Invoke-RestMethod http://127.0.0.1:11434/api/generate -Method Post -Body $body -ContentType 'application/json'
        $ps = (Invoke-RestMethod http://127.0.0.1:11434/api/ps).models | Where-Object { $_.name -eq $Model -or $_.name -eq "${Model}:latest" }   # exact: a wildcard would also match qwen3.5:4b-q8_0

        [pscustomobject]@{
            Model      = $Model
            Ctx        = $ctx
            NumGpu     = if ($gpu -ge 0) { $gpu } else { 'auto' }
            Run        = $_
            LoadS      = [math]::Round($r.load_duration / 1e9, 2)
            PromptTok  = $r.prompt_eval_count
            PromptTokS = [math]::Round($r.prompt_eval_count / ($r.prompt_eval_duration / 1e9))
            GenTok     = $r.eval_count
            GenTokS    = [math]::Round($r.eval_count / ($r.eval_duration / 1e9), 1)
            VramMiB    = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
            GpuPct     = if ($ps.size) { [int](100 * $ps.size_vram / $ps.size) } else { $null }
        }
    }
  }
}
Write-Progress -Activity "bench: $Model" -Completed
$results | Format-Table -AutoSize
