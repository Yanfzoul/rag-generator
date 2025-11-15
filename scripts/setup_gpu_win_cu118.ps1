<#
.SYNOPSIS
  One-time GPU setup for Windows (RTX 2080 / CUDA 11.8 wheels).

.DESCRIPTION
  - Creates a Python 3.11 virtual environment at .venv311
  - Installs CUDA 11.8 PyTorch wheels (torch/vision/audio)
  - Installs project requirements
  - Verifies CUDA availability

.USAGE
  1) Run in PowerShell (elevated if policy blocks):
       powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_win_cu118.ps1
  2) Activate venv:
       PowerShell: .\.venv311\Scripts\Activate.ps1
       CMD:        .venv311\Scripts\activate.bat
       Git Bash:   source .venv311/Scripts/activate
  3) Build on GPU:
       $env:RAG_CONFIG='config.test.yaml'; python ingest/build_index_hybrid_fast.py --gpu --batch 64
#>

$ErrorActionPreference = 'Stop'

function Exec([string]$cmd) {
  Write-Host "» $cmd" -ForegroundColor Cyan
  iex $cmd
}

# 1) Require Python 3.11 (CUDA wheels reliable on Win w/ 3.11)
try {
  $pyVersion = & py -3.11 -V 2>$null
  if (-not $pyVersion) { throw }
  Write-Host "Found: $pyVersion"
} catch {
  Write-Warning "Python 3.11 not found via 'py -3.11'. Install 3.11 from python.org."
  throw "Aborting setup (Python 3.11 required)."
}

# 2) Create venv .venv311
if (-not (Test-Path .venv311)) {
  Exec "py -3.11 -m venv .venv311"
} else {
  Write-Host ".venv311 already exists; reusing."
}

$venvPy = Join-Path (Resolve-Path .venv311) 'Scripts/python.exe'

# 3) Upgrade pip
Exec "`"$venvPy`" -m pip install --upgrade pip"

# 4) Install CUDA 11.8 PyTorch wheels
Exec "`"$venvPy`" -m pip install --index-url https://download.pytorch.org/whl/cu118 torch torchvision torchaudio"

# 5) Install project requirements
Exec "`"$venvPy`" -m pip install -r requirements.txt"

# Optional: BM25
try { Exec "`"$venvPy`" -m pip install rank_bm25" } catch { }

# Silence HF symlink warning
if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
  [System.Environment]::SetEnvironmentVariable('HF_HUB_DISABLE_SYMLINKS_WARNING','1','User')
  Write-Host "Set HF_HUB_DISABLE_SYMLINKS_WARNING=1 (User)"
}

# 6) Verify CUDA
Exec "`"$venvPy`" - << 'PY'
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('cuda_version:', torch.version.cuda)
    print('device:', torch.cuda.get_device_name(0))
else:
    raise SystemExit('ERROR: CUDA not available in this environment. Check NVIDIA driver and wheel.')
PY"

Write-Host "GPU setup complete (CUDA 11.8). Activate venv: .\\.venv311\\Scripts\\Activate.ps1" -ForegroundColor Green

