@echo off
cd /d "C:\Users\david\AppData\Roaming\Claude\Claude Extensions\ant.dir.cursortouch.windows-mcp"
echo === Checking venv ===
if exist .venv\Scripts\python.exe (
    echo venv exists
    .venv\Scripts\python.exe --version
) else (
    echo NO VENV FOUND
)
echo === Checking uv sync status ===
"C:\Users\david\.local\bin\uv.exe" sync 2>&1
echo === Checking if windows-mcp imports correctly ===
"C:\Users\david\.local\bin\uv.exe" run python -c "import windows_mcp; print('Import OK')" 2>&1
echo === Checking pywin32 and comtypes ===
"C:\Users\david\.local\bin\uv.exe" run python -c "import win32gui; import comtypes; import pyautogui; print('All deps OK')" 2>&1
echo === Running with verbose ===
"C:\Users\david\.local\bin\uv.exe" run --verbose windows-mcp 2>&1
