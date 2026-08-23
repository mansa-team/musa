# ponytail: delete year shards + corpus jsonls, keep pdfs/manifests — resume via materialized pdfs
$ErrorActionPreference = "Stop"
$root = Join-Path $PSScriptRoot "data"
if (-not (Test-Path $root)) { Write-Error "data dir not found: $root"; exit 1 }
$files = Get-ChildItem -Recurse -Path $root -Filter "*.jsonl" -ErrorAction SilentlyContinue
$count = ($files | Measure-Object).Count
Write-Output "found $count jsonls under $root"
$files | ForEach-Object { Write-Output "  removing $($_.FullName)"; $_.Delete() }
Write-Output "done — pdfs kept, manifests kept (scraper regenerates jsonls from cached pdfs via done - materialized)"
