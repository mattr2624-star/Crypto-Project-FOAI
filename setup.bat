@echo off
REM ============================================================================
REM Real-Time Crypto Volatility Detection Service - Windows Setup
REM ============================================================================
REM This batch file runs the PowerShell setup script with the correct permissions
REM
REM Usage: Double-click this file or run from command prompt
REM ============================================================================

echo.
echo ========================================
echo  Crypto Volatility Service Setup
echo ========================================
echo.

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: PowerShell is not installed or not in PATH
    echo Please install PowerShell and try again
    pause
    exit /b 1
)

REM Run the PowerShell setup script
powershell -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1"

if %ERRORLEVEL% neq 0 (
    echo.
    echo Setup encountered an error. Please check the output above.
    pause
    exit /b 1
)

pause

