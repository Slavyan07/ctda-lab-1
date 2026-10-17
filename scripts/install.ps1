$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot '..\install.bat') @args
exit $LASTEXITCODE
