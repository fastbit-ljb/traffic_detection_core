param(
    [switch]$NoBrowser,
    [switch]$UseSystemCuda,
    [ValidateSet('Auto', 'Cpu', 'Cuda')]
    [string]$InferenceRuntime = 'Auto'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendDir = Join-Path $projectRoot 'backend'
$frontendDir = Join-Path $projectRoot 'frontend'
$logsDir = Join-Path $projectRoot 'logs'
$apiUrl = 'http://127.0.0.1:8000'
$frontendUrl = 'http://127.0.0.1:5173'

function Test-ListeningPort {
    param([int]$Port)

    return $null -ne (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
}

function Wait-ForHealth {
    param(
        [string]$Url,
        [int]$MaxAttempts = 300
    )

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        try {
            $health = Invoke-RestMethod "$Url/health" -TimeoutSec 3
            if ($health.status) {
                return $health
            }
        } catch {
            if ($attempt % 15 -eq 0) {
                Write-Host "Backend is still starting ($attempt/$MaxAttempts). First startup may download and load YOLO weights..." -ForegroundColor Yellow
            }
            Start-Sleep -Seconds 1
        }
    }

    $errorLog = Join-Path $logsDir 'backend-dev.err.log'
    if (Test-Path $errorLog) {
        Write-Host "Backend error log:" -ForegroundColor Red
        Get-Content $errorLog -Tail 80
    }
    throw "Backend did not become healthy within $MaxAttempts seconds. Check logs/backend-dev.err.log."
}

function Get-ContentHash {
    param([string]$Path)

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Get-TorchRuntime {
    param(
        [ValidateSet('Auto', 'Cpu', 'Cuda')]
        [string]$Mode
    )

    if ($Mode -eq 'Cpu') {
        return [PSCustomObject]@{ Channel = 'cpu'; Description = 'CPU' }
    }

    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -eq $nvidiaSmi) {
        if ($Mode -eq 'Cuda') { throw '未检测到 nvidia-smi，无法安装 CUDA 推理运行时。' }
        return [PSCustomObject]@{ Channel = 'cpu'; Description = 'CPU（未检测到 NVIDIA 驱动）' }
    }

    $smiOutput = (& $nvidiaSmi.Source 2>$null | Out-String)
    if ($LASTEXITCODE -ne 0) {
        if ($Mode -eq 'Cuda') { throw 'nvidia-smi 无法执行，无法安装 CUDA 推理运行时。' }
        return [PSCustomObject]@{ Channel = 'cpu'; Description = 'CPU（NVIDIA 驱动不可用）' }
    }

    $cudaMatch = [regex]::Match($smiOutput, 'CUDA Version:\s*(?<version>\d+\.\d+)')
    if (-not $cudaMatch.Success) {
        if ($Mode -eq 'Cuda') { throw '无法从 NVIDIA 驱动读取支持的 CUDA 版本。' }
        return [PSCustomObject]@{ Channel = 'cpu'; Description = 'CPU（驱动未报告 CUDA 支持）' }
    }

    $cudaVersion = [version]$cudaMatch.Groups['version'].Value
    $channel = if ($cudaVersion -ge [version]'12.4') {
        'cu124'
    } elseif ($cudaVersion -ge [version]'12.1') {
        'cu121'
    } elseif ($cudaVersion -ge [version]'11.8') {
        'cu118'
    } else {
        $null
    }

    if ($null -eq $channel) {
        if ($Mode -eq 'Cuda') { throw "当前 NVIDIA 驱动仅支持 CUDA $cudaVersion，低于本项目的最低 CUDA 11.8 要求。" }
        return [PSCustomObject]@{ Channel = 'cpu'; Description = "CPU（驱动 CUDA $cudaVersion 低于 11.8）" }
    }

    $gpuNames = (& $nvidiaSmi.Source '--query-gpu=name' '--format=csv,noheader' 2>$null | Where-Object { $_.Trim() }) -join ', '
    $description = if ($gpuNames) { "NVIDIA $gpuNames，驱动 CUDA $cudaVersion" } else { "NVIDIA 驱动 CUDA $cudaVersion" }
    return [PSCustomObject]@{ Channel = $channel; Description = $description }
}

function Get-PythonWheelTag {
    param([string]$PythonExe)

    $wheelTag = (& $PythonExe -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}-cp{sys.version_info.major}{sys.version_info.minor}-win_amd64')").Trim()
    if ($LASTEXITCODE -ne 0 -or $wheelTag -notmatch '^cp(311|312)-cp\1-win_amd64$') {
        throw "Python runtime is unsupported by the bundled PyTorch version. Detected wheel tag: $wheelTag. Use Python 3.11 or 3.12."
    }

    return $wheelTag
}

function Download-FileWithResume {
    param(
        [string]$Url,
        [string]$Destination,
        [string]$Label
    )

    if (Test-Path -LiteralPath $Destination) {
        return
    }

    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -eq $curl) {
        throw 'curl.exe is required to download the CUDA runtime. It is included with supported Windows versions.'
    }

    $partialDownload = "$Destination.partial"
    Write-Host "Downloading $Label (supports resume; rerun this script after a network interruption)..."
    & $curl.Source `
        '--fail' '--location' '--continue-at' '-' `
        '--connect-timeout' '30' '--speed-limit' '1024' '--speed-time' '300' `
        '--retry' '8' '--retry-all-errors' '--retry-delay' '8' `
        '--output' $partialDownload $Url
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA runtime download was interrupted. The partial file was kept at $partialDownload; rerun the same command to resume."
    }

    Move-Item -LiteralPath $partialDownload -Destination $Destination -Force
}

function Install-TorchRuntime {
    param([string]$PythonExe)

    $runtime = Get-TorchRuntime -Mode $InferenceRuntime
    $runtimeStamp = Join-Path (Split-Path -Parent $PythonExe) '..\.traffic-torch-runtime.txt'
    $runtimeStamp = [IO.Path]::GetFullPath($runtimeStamp)
    $runtimeSignature = "torch=2.5.1;torchvision=0.20.1;channel=$($runtime.Channel)"
    $installedSignature = if (Test-Path -LiteralPath $runtimeStamp) {
        (Get-Content -LiteralPath $runtimeStamp -Raw).Trim()
    } else {
        ''
    }

    if ($installedSignature -eq $runtimeSignature) {
        return
    }

    Write-Host "Installing PyTorch runtime: $($runtime.Description) [$($runtime.Channel)]"
    # CUDA 通道用 curl 直下官方 wheel（R2 CDN，可断点续传）；CPU 通道的 pip 走清华 PyPI 镜像加速
    $torchIndex = "https://download.pytorch.org/whl/$($runtime.Channel)"
    if ($runtime.Channel -like 'cu*') {
        # CUDA wheels are several gigabytes. Downloading with curl preserves partial data,
        # unlike pip's temporary download, so a failed network transfer can be resumed.
        $wheelTag = Get-PythonWheelTag -PythonExe $PythonExe
        $downloadDir = Join-Path (Split-Path -Parent (Split-Path -Parent $PythonExe)) '.traffic-downloads'
        New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null

        $torchWheel = Join-Path $downloadDir "torch-2.5.1+$($runtime.Channel)-$wheelTag.whl"
        $torchvisionWheel = Join-Path $downloadDir "torchvision-0.20.1+$($runtime.Channel)-$wheelTag.whl"
        Download-FileWithResume -Url "$torchIndex/torch-2.5.1%2B$($runtime.Channel)-$wheelTag.whl" -Destination $torchWheel -Label 'PyTorch CUDA runtime (about 2.5 GB)'
        Download-FileWithResume -Url "$torchIndex/torchvision-0.20.1%2B$($runtime.Channel)-$wheelTag.whl" -Destination $torchvisionWheel -Label 'TorchVision CUDA runtime'

        & $PythonExe -m pip install --upgrade --force-reinstall --no-cache-dir --timeout 1200 --retries 5 $torchWheel $torchvisionWheel | Out-Host
        if ($LASTEXITCODE -eq 0) {
            Remove-Item -LiteralPath $torchWheel, $torchvisionWheel -Force -ErrorAction SilentlyContinue
        }
    } else {
        # Windows 的 PyPI torch 就是 CPU 构建（CUDA 版只在 download.pytorch.org 的 +cu 通道），走清华源国内直连最快
        & $PythonExe -m pip install --upgrade --force-reinstall --no-cache-dir --timeout 1200 --retries 5 'torch==2.5.1' 'torchvision==0.20.1' --index-url 'https://pypi.tuna.tsinghua.edu.cn/simple' | Out-Host
    }
    if ($LASTEXITCODE -ne 0) { throw 'Failed to install the PyTorch runtime.' }
    Set-Content -LiteralPath $runtimeStamp -Value $runtimeSignature -NoNewline
}

function Install-BackendDependencies {
    # Auto mode reuses a complete system CUDA environment when available. This
    # avoids downloading another multi-gigabyte PyTorch wheel on developer PCs.
    if ($UseSystemCuda -or $InferenceRuntime -eq 'Auto') {
        $systemPython = (Get-Command python -ErrorAction Stop).Source
        $cudaDetails = (& $systemPython -c "import torch, fastapi, uvicorn, ultralytics, cv2; assert torch.cuda.is_available(); print(f'{torch.__version__}; CUDA {torch.version.cuda}; {torch.cuda.get_device_name(0)}')" 2>&1 | Out-String).Trim()
        if ($LASTEXITCODE -eq 0) {
            $env:TRAFFIC_ENABLE_GPU_ACCELERATION = 'true'
            Write-Host "Using existing system CUDA runtime: $cudaDetails"
            return $systemPython
        }
        if ($UseSystemCuda) {
            throw "System CUDA Python cannot run this project. $cudaDetails"
        }
    }

    $venvDirectory = Join-Path $backendDir '.venv'
    $venvPython = Join-Path $venvDirectory 'Scripts\python.exe'
    $requirementsFile = Join-Path $backendDir 'requirements.txt'
    $requirementsStamp = Join-Path $venvDirectory '.traffic-requirements.sha256'
    $requirementsHash = Get-ContentHash $requirementsFile
    $installedHash = if (Test-Path -LiteralPath $requirementsStamp) {
        (Get-Content -LiteralPath $requirementsStamp -Raw).Trim()
    } else {
        ''
    }

    if (-not (Test-Path -LiteralPath $venvPython)) {
        $systemPython = (Get-Command python -ErrorAction Stop).Source
        Write-Host 'Creating backend virtual environment...'
        & $systemPython -m venv $venvDirectory
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create backend virtual environment.' }
    }

    Install-TorchRuntime -PythonExe $venvPython

    if ($installedHash -ne $requirementsHash) {
        Write-Host 'Installing backend dependencies...'
        & $venvPython -m pip install --upgrade pip | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Failed to upgrade pip.' }
        & $venvPython -m pip install -r $requirementsFile --index-url 'https://pypi.tuna.tsinghua.edu.cn/simple' | Out-Host
        if ($LASTEXITCODE -ne 0) { throw 'Failed to install backend dependencies.' }
        Set-Content -LiteralPath $requirementsStamp -Value $requirementsHash -NoNewline
    }

    return $venvPython
}

function Install-FrontendDependencies {
    $nodeModules = Join-Path $frontendDir 'node_modules'
    $lockFile = Join-Path $frontendDir 'package-lock.json'
    $lockStamp = Join-Path $nodeModules '.traffic-package-lock.sha256'
    $lockHash = Get-ContentHash $lockFile
    $installedHash = if (Test-Path -LiteralPath $lockStamp) {
        (Get-Content -LiteralPath $lockStamp -Raw).Trim()
    } else {
        ''
    }

    if ($installedHash -ne $lockHash) {
        $npm = (Get-Command npm -ErrorAction Stop).Source
        Write-Host 'Installing frontend dependencies...'
        Push-Location $frontendDir
        try {
            & $npm ci | Out-Host
            if ($LASTEXITCODE -ne 0) { throw 'Failed to install frontend dependencies.' }
        } finally {
            Pop-Location
        }
        Set-Content -LiteralPath $lockStamp -Value $lockHash -NoNewline
    }
}

if (-not (Test-Path (Join-Path $backendDir 'app\main.py'))) {
    throw "Run this script from the project root. Backend entry point was not found."
}

if (-not (Test-Path (Join-Path $frontendDir 'package.json'))) {
    throw "Frontend package.json was not found."
}

New-Item -ItemType Directory -Force -Path $logsDir | Out-Null
$pythonExe = Install-BackendDependencies
Install-FrontendDependencies

& $pythonExe -c "import fastapi, uvicorn, ultralytics, cv2"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependencies are missing. Install backend/requirements.txt first."
}

if (Test-ListeningPort 8000) {
    Write-Host "Backend already listening on port 8000."
} else {
    $backendLog = Join-Path $logsDir 'backend-dev.log'
    $backendErrorLog = Join-Path $logsDir 'backend-dev.err.log'
    $backend = Start-Process -FilePath $pythonExe `
        -ArgumentList '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8000' `
        -WorkingDirectory $backendDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $backendLog `
        -RedirectStandardError $backendErrorLog `
        -PassThru
    Write-Host "Backend started. PID: $($backend.Id)"
}

$health = Wait-ForHealth $apiUrl

if (Test-ListeningPort 5173) {
    Write-Host "Frontend already listening on port 5173."
} else {
    $frontendLog = Join-Path $logsDir 'frontend-dev.log'
    $frontendErrorLog = Join-Path $logsDir 'frontend-dev.err.log'
    $frontendCommand = 'set "VITE_API_BASE_URL=http://127.0.0.1:8000" && npm run dev -- --host 127.0.0.1 --port 5173'
    $frontend = Start-Process -FilePath $env:ComSpec `
        -ArgumentList '/c', $frontendCommand `
        -WorkingDirectory $frontendDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $frontendLog `
        -RedirectStandardError $frontendErrorLog `
        -PassThru
    Write-Host "Frontend started. PID: $($frontend.Id)"
}

Write-Host "Project is ready."
Write-Host "Frontend: $frontendUrl"
Write-Host "API docs:  $apiUrl/docs"
Write-Host "Health:    $($health.status)"

if (-not $NoBrowser) {
    Start-Process $frontendUrl
}
