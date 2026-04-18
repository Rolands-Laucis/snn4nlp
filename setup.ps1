#Requires -Version 7
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$VenvDir = ".venv"

Write-Host "Creating virtual environment..."
python -m venv $VenvDir

Write-Host "Activating..."
& "$VenvDir\Scripts\Activate.ps1"

# pip freeze | Out-File -FilePath "req.txt" -Encoding utf8

Write-Host "Verifying install from req.txt..."
pip install -r req.txt

Write-Host "Done. Environment ready."