$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '..\check.bat') @args
exit $LASTEXITCODE
