$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '..\start.bat') @args
exit $LASTEXITCODE
