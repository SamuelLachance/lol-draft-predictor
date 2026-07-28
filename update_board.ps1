# Daily board update — run from YOUR machine (residential IP, where Oracle's Elixir's
# Google Drive works; GitHub Actions datacenter IPs are quota-blocked on that file).
# It: (1) pulls the freshest OE data (2026 + 2025), (2) merges into the snapshot,
# (3) rebuilds all ratings, (4) deploys to gh-pages, (5) updates the shared Release
# so the cloud CI stays in sync. Schedule it once (see the schtasks line at the bottom).
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Admin\Documents\lolcoach\draftwork"

Write-Host "[1/4] Refreshing snapshot from Oracle's Elixir + updating the Release..."
python ml/refresh_snapshot.py --upload      # best-effort: skips a year if Drive quota blocks it

Write-Host "[2/4] Preparing build input..."
New-Item -ItemType Directory -Force data/processed | Out-Null
Copy-Item data/board_input.parquet data/processed/oe_all.parquet -Force

Write-Host "[3/4] Rebuilding ratings..."
python ml/build_drafts.py
python ml/build_drafts_team.py
python ml/build_board.py

Write-Host "[4/4] Deploying to gh-pages..."
Push-Location web/public
if (Test-Path .git) { Remove-Item -Recurse -Force .git }
git init -q
git checkout -q -b gh-pages
git config user.name "SamuelLachance"
git config user.email "samuellachance5@gmail.com"
"" | Out-File .nojekyll -Encoding ascii
git add -A
git commit -q -m "Local board update $(Get-Date -Format yyyy-MM-dd)"
git push -q --force "https://github.com/SamuelLachance/lol-draft-predictor.git" gh-pages
if (Test-Path .git) { Remove-Item -Recurse -Force .git }
Pop-Location

$meta = Get-Content web/public/data/meta.json | ConvertFrom-Json
Write-Host ("Done. Data through {0} ({1} games)." -f $meta.data_through, $meta.n_games)

# --- Schedule this to run every day at 08:00 (run once in a terminal) ---
# schtasks /Create /SC DAILY /ST 08:00 /TN "LoL board update" /TR "powershell -NoProfile -ExecutionPolicy Bypass -File C:\Users\Admin\Documents\lolcoach\draftwork\update_board.ps1"
