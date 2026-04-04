@echo off

echo Cleaning build and dist directories...
rd /q /s build 2>nul
rd /q /s dist 2>nul

echo Configuring CMake...
cmake -B build -DBUILD_PORTABLE=ON -DBUILD_PYTHON_BINDINGS=ON -DBUILD_PYTHON_STUBS=ON
if %errorlevel% neq 0 (
    echo CMake configuration failed!
    pause
    exit /b %errorlevel%
)

echo Building with 16 parallel jobs...
cmake --build build -j 16 --config Release
if %errorlevel% neq 0 (
    echo Build failed! Check build.log for details.
    pause
    exit /b %errorlevel%
)

echo Installing to dist...
cmake --install build --prefix ./dist
if %errorlevel% neq 0 (
    echo Install failed!
    pause
    exit /b %errorlevel%
)

echo.
echo Build completed successfully!
pause
