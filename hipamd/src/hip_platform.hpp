/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */
#pragma once

#include "hip_internal.hpp"
#include "hip_fatbin.hpp"
#include "device/device.hpp"
#include "hip_code_object.hpp"

namespace hip_impl {

hipError_t ihipOccupancyMaxActiveBlocksPerMultiprocessor(
    int* maxBlocksPerCU, int* numBlocksPerGrid, int* bestBlockSize, const amd::Device& device,
    hipFunction_t func, int inputBlockSize, size_t dynamicSMemSize, bool bCalcPotentialBlkSz);
}  // namespace hip_impl

// Unique file descriptor class
struct UniqueFD {
  UniqueFD(const std::string& fpath, amd::Os::FileDesc fdesc, size_t fsize)
           : fpath_(fpath), fdesc_(fdesc), fsize_(fsize) {}

  const std::string fpath_;        //!< File path of this unique file
  const amd::Os::FileDesc fdesc_;  //!< File Descriptor
  const size_t fsize_;             //!< File Size
};

namespace hip {
class PlatformState {
  amd::Monitor lock_{"Guards PlatformState globals", true};

  // global level lock for unique file descritor map: ufd_map_
  amd::Monitor ufd_lock_{"Unique FD Store Lock", true};

  // Singleton object
  static PlatformState* platform_;
  PlatformState() {}
  ~PlatformState() {}

 public:
  void init();

  // Dynamic Code Objects functions
  hipError_t loadModule(hipModule_t* module, const char* fname, const void* image = nullptr);
  hipError_t unloadModule(hipModule_t hmod);

  hipError_t getDynFunc(hipFunction_t* hfunc, hipModule_t hmod, const char* func_name);
  hipError_t getDynGlobalVar(const char* hostVar, hipModule_t hmod, hipDeviceptr_t* dev_ptr,
                             size_t* size_ptr);
  hipError_t getDynTexRef(const char* hostVar, hipModule_t hmod, textureReference** texRef);

  hipError_t registerTexRef(textureReference* texRef, hipModule_t hmod, std::string name);
  hipError_t getDynTexGlobalVar(textureReference* texRef, hipDeviceptr_t* dev_ptr,
                                size_t* size_ptr);

  // Singleton instance
  static PlatformState& instance() {
    if (platform_ == nullptr) {
      // __hipRegisterFatBinary() will call this when app starts, thus
      // there is no multiple entry issue here.
      platform_ = new PlatformState();
    }
    return *platform_;
  }

  // Static Code Objects functions
  hip::FatBinaryInfo** addFatBinary(const void* data);
  hipError_t removeFatBinary(hip::FatBinaryInfo** module);
  hipError_t digestFatBinary(const void* data, hip::FatBinaryInfo*& programs);

  hipError_t registerStatFunction(const void* hostFunction, hip::Function* func);
  hipError_t registerStatGlobalVar(const void* hostVar, hip::Var* var);
  hipError_t registerStatManagedVar(hip::Var* var);

  const char* getStatFuncName(const void* hostFunction);
  hipError_t getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId);
  hipError_t getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                              size_t* size_ptr);

  hipError_t initStatManagedVarDevicePtr(int deviceId);

  // Load hip dynamic library
  void* getDynamicLibraryHandle();
  void setDynamicLibraryHandle(void* handle);

  // Exec Functions
  void setupArgument(const void* arg, size_t size, size_t offset);
  void configureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);
  void popExec(ihipExec_t& exec);

  std::shared_ptr<UniqueFD> GetUniqueFileHandle(const std::string& file_path);
  bool CloseUniqueFileHandle(const std::shared_ptr<UniqueFD>& ufd);

  size_t UfdMapSize() const { return ufd_map_.size(); }

 private:
  // Dynamic Code Object map, keyin module to get the corresponding object
  std::unordered_map<hipModule_t, hip::DynCO*> dynCO_map_;
  hip::StatCO statCO_;  //!< Static Code object var
  bool initialized_{false};
  std::unordered_map<textureReference*, std::pair<hipModule_t, std::string>> texRef_map_;

  std::unordered_map<std::string, std::shared_ptr<UniqueFD>> ufd_map_; //!< Unique File Desc Map

  void* dynamicLibraryHandle_{nullptr};

  // For tracking dump requests during initialization
  struct FunctionDumpInfo {
      std::string deviceName;
      hip::FatBinaryInfo** modules;
  };
  std::vector<FunctionDumpInfo> pendingFunctionDumps_;
  void processPendingDumps();
  public:
  // Dump functions for debugging
  hipError_t dumpFatBinary(const char* deviceName, hip::FatBinaryInfo** modules, const char* dumpDir = nullptr);
  void queueFunctionForDump(hip::FatBinaryInfo** modules, const char* deviceName);
  bool isInitialized() const { return initialized_; }

  // Class to manage fat binary swaps
  class SwapManager {
  public:
      void queueSwap(hip::FatBinaryInfo** modules, const char* deviceName, const char* filePath) {
          pendingSwaps_.push_back({deviceName, filePath, modules});
      }

      // Returns swapped binary if one exists, otherwise returns original
      const void* getSwappedBinary(const void* original) const {
          auto it = swapMap_.find(original);
          if (it != swapMap_.end())
            fprintf(stderr, "fat binary %p is swapped by %p\n", original, it->second);
          return (it != swapMap_.end()) ? it->second : original;
      }

      // Build swap map from pending swaps, using modules map to find matches
      void processSwaps(const std::unordered_map<const void*, FatBinaryInfo*>& modules) {
          for (const auto& swap : pendingSwaps_) {
              for (auto& it : modules) {
                  if (it.second == *swap.modules) {
                      fprintf(stderr, "[DEBUG] Found matching module for function %s\n", 
                              swap.functionName.c_str());

                      const void* new_image = loadSwapFile(swap.filePath.c_str());
                      if (new_image != nullptr) {
                          fprintf(stderr, "[DEBUG] Will swap fat binary for %s\n", 
                                  swap.functionName.c_str());
                          swapMap_[it.first] = new_image;
                      }
                      break;
                  }
              }
          }
          pendingSwaps_.clear();
      }

  private:
      struct SwapInfo {
          std::string functionName;
          std::string filePath;
          hip::FatBinaryInfo** modules;
      };
      std::vector<SwapInfo> pendingSwaps_;
      std::unordered_map<const void*, const void*> swapMap_;

      const void* loadSwapFile(const char* filePath) {
          amd::Os::FileDesc fdesc;
          size_t fsize = 0;
          if (!amd::Os::GetFileHandle(filePath, &fdesc, &fsize)) {
              fprintf(stderr, "[DEBUG] Failed to open swap file: %s\n", filePath);
              return nullptr;
          }

          const void* new_image = nullptr;
          if (!amd::Os::MemoryMapFileDesc(fdesc, fsize, 0, &new_image)) {
              fprintf(stderr, "[DEBUG] Failed to map swap file content\n");
              amd::Os::CloseFileHandle(fdesc);
              return nullptr;
          }
          fprintf(stderr, "[DEBUG] mapped file %s for swapping\n", filePath);
          return new_image;
      }
  };

  SwapManager swapManager_;

public:
    void queueFunctionForSwap(hip::FatBinaryInfo** modules, const char* deviceName, 
                             const char* filePath) {
        swapManager_.queueSwap(modules, deviceName, filePath);
    }
};
}  // namespace hip
