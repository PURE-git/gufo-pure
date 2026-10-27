$ErrorActionPreference = "Stop"

$rocm = "C:/Users/Naessah/AppData/Local/Gufo/dependencies/therock-10.0.0"
$vcpkg_bin = "D:/AI_Stack/Engines/Windows/thomas9120/gufo-windows/build/vcpkg_installed/x64-windows/bin"

$env:PATH = "$rocm/bin;$vcpkg_bin;$env:PATH"

$exe = "build/test/tests/models/ornith122b/ornith122b_load_test.exe"
$model = "D:\AI_Stack\Models\Harsha001\Ornith-1.5-122B-ROCmFPX\Ornith-1.5-122B-ROCmFP4-Q4_0_STRIX-00001-of-00002.gguf"

Write-Host "Executing $exe on $model..."
& $exe $model
