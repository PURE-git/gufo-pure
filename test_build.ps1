$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
Write-Host "Loading MSVC environment from $vcvars..."
$envLines = & cmd.exe /c "`"$vcvars`" >nul && set"
foreach ($line in $envLines) {
    if ($line -match "^([^=]+)=(.*)$") {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

$cmake = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$ninja = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
$rocm = "C:/Users/Naessah/AppData/Local/Gufo/dependencies/therock-10.0.0"
$vcpkg = "C:/Users/Naessah/AppData/Local/Gufo/dependencies/vcpkg-9e2895bf6afb"
$clang = "$rocm/lib/llvm/bin/clang.exe"
$clangxx = "$rocm/lib/llvm/bin/clang++.exe"
$vcpkg_installed = "D:/AI_Stack/Engines/Windows/thomas9120/gufo-windows/build/vcpkg_installed"

$env:PATH = "$rocm/bin;$rocm/lib/llvm/bin;$env:PATH"
$env:HIP_PATH = $rocm
$env:HIP_DEVICE_LIB_PATH = "$rocm/lib/llvm/amdgcn/bitcode"

Write-Host "Configuring CMake for ornith122b..."
& $cmake -S . -B build/test `
    -G "Ninja" `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_C_COMPILER=$clang" `
    "-DCMAKE_CXX_COMPILER=$clangxx" `
    "-DCMAKE_TOOLCHAIN_FILE=$vcpkg/scripts/buildsystems/vcpkg.cmake" `
    "-DVCPKG_TARGET_TRIPLET=x64-windows" `
    "-DVCPKG_INSTALLED_DIR=$vcpkg_installed" `
    "-DCMAKE_BUILD_TYPE=RelWithDebInfo" `
    "-DENGINE_ENABLE_HIP=ON" `
    "-DCMAKE_HIP_COMPILER=$clangxx" `
    "-DCMAKE_PREFIX_PATH=$rocm" `
    "-DBUILD_TESTING=ON"

if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure failed"
    exit 1
}

Write-Host "Building ornith122b_load_test..."
& $cmake --build build/test --target ornith122b_load_test --parallel 4

if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed"
    exit 1
}

Write-Host "Build succeeded!"
