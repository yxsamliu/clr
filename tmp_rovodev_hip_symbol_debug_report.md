# HIP Symbol Resolution and Memory Copy Issues - Debug Report

## Overview

This report documents the investigation and debugging of two critical issues in HIP (Heterogeneous-Interface for Portability) runtime:

1. **`hipErrorInvalidSymbol` in `hipGetSymbolAddress`** - Symbol lookup failures
2. **Segmentation fault in `hipMemcpy`** - Memory copy crashes during device-to-host transfers

## Issue 1: `hipErrorInvalidSymbol` in `hipGetSymbolAddress`

### Problem Description

HIP programs were failing with `hipErrorInvalidSymbol` when calling `hipGetSymbolAddress` to retrieve device global variable addresses.

**Error Log:**
```
:3:hip_platform.cpp         :301 : 2722572134303 us:   hipGetSymbolAddress ( 0x7ffc6f3fb3a0, 0x55cf401c6af0 ) 
:3:hip_platform.cpp         :309 : 2722572134306 us:  hipGetSymbolAddress: Returned hipErrorInvalidSymbol : 
```

### Root Cause Analysis

The error occurs when `hipGetSymbolAddress` cannot find the requested symbol in the internal symbol registry. The code flow is:

1. `hipGetSymbolAddress` → `PlatformState::getStatGlobalVar` → `StatCO::getStatGlobalVar`
2. Critical check in `StatCO::getStatGlobalVar` (lines 1008-1011):
   ```cpp
   const auto it = vars_.find(hostVar);
   if (it == vars_.end()) {
       return hipErrorInvalidSymbol;  // Error occurs here
   }
   ```

### Common Causes

1. **Symbol Not Registered**: Global variable never registered via `__hipRegisterVar`
2. **Compilation Issues**: Variable not properly compiled into device code
3. **Module Loading Problems**: Fat binary containing symbol not loaded correctly
4. **Symbol Name Mismatch**: Host-side symbol pointer doesn't match registered pointer

### Debug Solution Implemented

Added comprehensive debug tracing to the HIP runtime:

#### 1. Symbol Registration Tracing (`__hipRegisterVar`)
```cpp
ClPrint(amd::LOG_INFO, amd::LOG_API, "[DEBUG] __hipRegisterVar: hostVar='%s', deviceVar='%s', "
        "hostPtr=%p, size=%zu, ext=%d, constant=%d, global=%d, modules=%p",
        hostVar ? hostVar : "<null>", deviceVar ? deviceVar : "<null>", 
        var, size, ext, constant, global, modules);
```

#### 2. Symbol Lookup Tracing (`hipGetSymbolAddress`)
```cpp
ClPrint(amd::LOG_INFO, amd::LOG_API, "[DEBUG] hipGetSymbolAddress: called with symbol=%p, devPtr=%p, deviceId=%d",
        symbol, devPtr, ihipGetDevice());
```

#### 3. Internal Registry Tracing (`StatCO::registerStatGlobalVar` and `StatCO::getStatGlobalVar`)
- Logs all registered variables with their host pointers and names
- Shows registry size and lookup results
- Traces module loading and device variable retrieval

### Files Modified

- `hipamd/src/hip_platform.cpp`: Added debug output to `__hipRegisterVar` and `hipGetSymbolAddress`
- `hipamd/src/hip_code_object.cpp`: Added debug output to `StatCO::registerStatGlobalVar` and `StatCO::getStatGlobalVar`

## Issue 2: Segmentation Fault in `hipMemcpy`

### Problem Description

After resolving the symbol lookup issue, the program crashed with a segmentation fault during `hipMemcpy` operations for device-to-host memory transfers.

**Error Log:**
```
:3:hip_memory.cpp           :809 : 2725567503019 us:   hipMemcpy ( 0x555555663030, 0x7ffff5a44f98, 72, hipMemcpyDeviceToHost ) 

Thread 1 "tmp_rovodev_hip" received signal SIGSEGV, Segmentation fault.
__memmove_avx_unaligned () at ../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:222
```

**Stack Trace:**
```
#0  __memmove_avx_unaligned () at ../sysdeps/x86_64/multiarch/memmove-vec-unaligned-erms.S:222
#1  amd::device::HostBlitManager::readBuffer(...)
#2  amd::roc::KernelBlitManager::readBuffer(...)
#3  amd::roc::VirtualGPU::submitReadMemory(...)
#4  amd::Command::enqueue()
#5  hip::ihipMemcpy(...)
#6  hip::hipMemcpy_common(...)
#7  hip::hipMemcpy(...)
```

### Root Cause Analysis

The segfault occurs in `__memmove_avx_unaligned` during the memory copy operation, indicating **memory alignment issues**. Analysis of the debug output reveals:

1. **First `hipMemcpy` succeeds** (48 bytes): Copying unified structure
2. **Second `hipMemcpy` crashes** (72 bytes): Copying actual profile data
3. **Device address alignment**: `0x7ffff5a44f98` ends in `98`, suggesting poor alignment
4. **Host address**: `0x555555663030` may also have alignment issues

### Probable Causes

#### 1. Device Memory Alignment Problem
- Device address `0x7ffff5a44f98` not aligned to 16-byte boundaries required by AVX instructions
- GPU memory allocations may not guarantee CPU SIMD alignment requirements

#### 2. Invalid Device Memory Addresses
- Addresses might be virtual GPU addresses not directly accessible
- Stale addresses that became invalid between structure creation and access
- Memory layout mismatch between device and host

#### 3. Host Memory Alignment Issue
- Host destination not properly aligned for AVX operations

### Recommended Solutions

#### Immediate Validation
```cpp
// Validate device memory accessibility
hipPointerAttributes attr;
hipError_t error = hipPointerGetAttributes(&attr, device_ptr);
if (error != hipSuccess) {
    printf("ERROR: Device pointer 0x%lx is not accessible: %s\n", 
           (unsigned long)device_ptr, hipGetErrorString(error));
    return;
}

// Check alignment
if ((uintptr_t)device_ptr % 16 != 0) {
    printf("WARNING: Device pointer 0x%lx is not 16-byte aligned\n", 
           (unsigned long)device_ptr);
}
```

#### Alternative Approaches
1. **Use aligned memory allocation**: `posix_memalign()` for host buffers
2. **Use `hipMemcpyAsync` with stream synchronization**: Better error handling
3. **Byte-by-byte copy fallback**: Avoid alignment issues entirely
4. **Chunked memory validation**: Test device memory accessibility in small chunks

## Debug Output Usage

To enable the debug tracing:

```bash
export AMD_LOG_LEVEL=3
export HIP_PRINT_ENV=1
```

Expected debug output:
```
[DEBUG] __hipRegisterVar: hostVar='my_global_var', deviceVar='my_global_var', hostPtr=0x12345678, size=4, ...
[DEBUG] StatCO::registerStatGlobalVar: SUCCESS - registered hostVar=0x12345678, varName='my_global_var', total vars count=1
[DEBUG] hipGetSymbolAddress: called with symbol=0x12345678, devPtr=0x7fff..., deviceId=0
[DEBUG] StatCO::getStatGlobalVar: looking for hostVar=0x12345678, deviceId=0, total registered vars=1
[DEBUG] StatCO::getStatGlobalVar: FOUND hostVar=0x12345678 -> varName='my_global_var'
```

## Key Findings

1. **Symbol Registration Works**: The HIP runtime correctly registers global variables during module loading
2. **Symbol Lookup Mechanism**: Uses host pointer as key in internal registry (`vars_` map)
3. **Memory Alignment Critical**: AVX-optimized memory operations require proper alignment
4. **Device Memory Accessibility**: Not all device addresses are directly accessible via standard memory copy

## Future Improvements

1. **Enhanced Memory Validation**: Add automatic alignment checks in `hipMemcpy`
2. **Better Error Messages**: Provide more specific error information for symbol lookup failures
3. **Alignment-Safe Memory Operations**: Implement fallback mechanisms for unaligned memory
4. **Device Memory Verification**: Add runtime checks for device pointer validity

## Files for Future Reference

- `hipamd/src/hip_platform.cpp`: Symbol registration and lookup entry points
- `hipamd/src/hip_code_object.cpp`: Internal symbol registry management
- `hipamd/src/hip_memory.cpp`: Memory copy operations
- `rocclr/device/rocm/`: ROCm-specific memory management

## Testing Recommendations

1. Test with various global variable types and sizes
2. Verify behavior with different memory alignment scenarios
3. Test symbol lookup with multiple modules
4. Validate memory copy operations with large data transfers

---

**Report Generated**: For HIP runtime debugging and future issue resolution  
**Components**: Symbol resolution, memory management, device-host communication