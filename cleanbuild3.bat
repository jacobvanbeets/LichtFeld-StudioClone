git pull
rd /q /s build
rd /q /s dist
set PATH=C:\Strawberry\perl\bin;%PATH%
cmake -B build -DBUILD_PORTABLE=ON -DBUILD_PYTHON_BINDINGS=ON -DBUILD_PYTHON_STUBS=ON
cmake --build build -j 16 --config Release
cmake --install build --prefix ./dist`
