<#
.SYNOPSIS
  One-time GPU setup for this project on Windows with an RTX 2080 (CUDA).

.DESCRIPTION
  - Creates a Python 3.11 virtual environment at .venv311
  - Installs CUDA-enabled PyTorch (cu121) + torchvision/torchaudio
  - Installs project requirements
  - Verifies CUDA availability

.USAGE
  1) Right-click this file > Run with PowerShell, or run from an elevated
     PowerShell window:  powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_win.ps1
  2) Then activate the venv in your shell:
       PowerShell:   .\.venv311\Scripts\Activate.ps1
       CMD:          .venv311\Scripts\activate.bat
       Git Bash:     source .venv311/Scripts/activate
  3) Build on GPU, e.g.:
       $env:RAG_CONFIG='config.test.yaml'; python ingest/build_index_hybrid_fast.py --gpu --batch 64

.NOTES
  - Requires Python 3.11 to be installed (https://www.python.org/downloads/windows/)
  - Uses CUDA 12.1 wheels; compatible with RTX 2080 and recent NVIDIA drivers.
  - You do NOT need to install the CUDA Toolkit; wheels bundle CUDA.
#>

param()

$ErrorActionPreference = 'Stop'

function Exec([string]$cmd) {
  Write-Host "» $cmd" -ForegroundColor Cyan
  iex $cmd
}

# 1) Locate Python 3.11
try {
  $pyVersion = & py -3.11 -V 2>$null
  if (-not $pyVersion) { throw }
  Write-Host "Found: $pyVersion"
} catch {
  Write-Warning "Python 3.11 not found via 'py -3.11'. Please install Python 3.11 for best CUDA wheel support."
  Write-Warning "Download: https://www.python.org/downloads/release/python-3110/"
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

# 4) Install CUDA-enabled PyTorch (CUDA 12.1 wheels)
#    Install torch + torchvision first; torchaudio is optional and may not have wheels for all combos.
Exec "`"$venvPy`" -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision"
try {
  Exec "`"$venvPy`" -m pip install --index-url https://download.pytorch.org/whl/cu121 torchaudio"
} catch {
  Write-Warning "torchaudio wheel not found for this environment; skipping (not required for this project)."
}

# 5) Install project requirements
Exec "`"$venvPy`" -m pip install -r requirements.txt"

# Optional: BM25 for lexical scoring
try { Exec "`"$venvPy`" -m pip install rank_bm25" } catch { }

# Optional: silence HF symlink warning on Windows
if (-not $env:HF_HUB_DISABLE_SYMLINKS_WARNING) {
  [System.Environment]::SetEnvironmentVariable('HF_HUB_DISABLE_SYMLINKS_WARNING','1','User')
  Write-Host "Set HF_HUB_DISABLE_SYMLINKS_WARNING=1 (User)"
}

# 6) Verify CUDA availability
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

Write-Host "GPU setup complete. Activate venv: .\\.venv311\\Scripts\\Activate.ps1" -ForegroundColor Green
