/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

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

#include <hip/hip_runtime.h>

#include "hip_event.hpp"
#if !defined(_MSC_VER)
#include <unistd.h>
#endif

namespace hip {

// Guards global event set
static amd::Monitor eventSetLock{};
static std::unordered_set<hipEvent_t> eventSet;

bool Event::ready() {
  if (event_->status() != CL_COMPLETE) {
    event_->notifyCmdQueue();
  }
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  bool ready = CheckHwEvent();
  if (!ready) {
    ready = (event_->status() == CL_COMPLETE);
  }
  return ready;
}

bool EventDD::ready() {
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  bool ready = CheckHwEvent();
  // FIXME: Remove status check entirely
  if (!ready) {
    ready = (event_->status() == CL_COMPLETE);
  }
  return ready;
}

hipError_t Event::query() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  return ready() ? hipSuccess : hipErrorNotReady;
}

// ================================================================================================
hipError_t Event::synchronize() {
  amd::ScopedLock lock(lock_);

  // If event is not recorded, event_ is null, hence return hipSuccess
  if (event_ == nullptr) {
    return hipSuccess;
  }

  auto hip_device = g_devices[deviceId()];
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  static constexpr bool kWaitCompletion = true;
  if (!hip_device->devices()[0]->IsHwEventReady(*event_, kWaitCompletion, flags_)) {
    event_->awaitCompletion();
  }
  return hipSuccess;
}

// ================================================================================================
bool Event::awaitEventCompletion() {
  return event_->awaitCompletion();
}

bool EventDD::awaitEventCompletion() {
  return g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, true, flags_);
}

hipError_t Event::elapsedTime(Event& eStop, float& ms) {
  amd::ScopedLock startLock(lock_);
  if (this == &eStop) {
    ms = 0.f;
    if (event_ == nullptr) {
      return hipErrorInvalidHandle;
    }

    if (flags_ & hipEventDisableTiming) {
      return hipErrorInvalidHandle;
    }

    if (!ready()) {
      return hipErrorNotReady;
    }

    return hipSuccess;
  }
  amd::ScopedLock stopLock(eStop.lock());

  if (event_ == nullptr || eStop.event() == nullptr) {
    return hipErrorInvalidHandle;
  }

  if ((flags_ | eStop.flags_) & hipEventDisableTiming) {
    return hipErrorInvalidHandle;
  }

  if (!ready() || !eStop.ready()) {
    return hipErrorNotReady;
  }

  if (event_ == eStop.event_) {
    // Events are the same, which indicates the stream is empty and likely
    // eventRecord is called on another stream. For such cases insert and measure a
    // marker.
    amd::Command* command = new amd::Marker(*event_->command().queue(), kMarkerDisableFlush);
    command->enqueue();
    command->awaitCompletion();
    ms = static_cast<float>(static_cast<int64_t>(command->event().profilingInfo().end_) - time(false)) /
        1000000.f;
    command->release();
  } else {
    // Note: with direct dispatch eStop.ready() relies on HW event, but CPU status can be delayed.
    // Hence for now make sure CPU status is updated by calling awaitCompletion();
    awaitEventCompletion();
    eStop.awaitEventCompletion();
    if (unrecorded_ && eStop.isUnRecorded()) {
      // Both the events are not recorded, just need the end and start of stop event
      ms = static_cast<float>(eStop.time(false) - eStop.time(true)) / 1000000.f;
    } else {
      ms = static_cast<float>(eStop.time(false) - time(false)) / 1000000.f;
    }
  }
  return hipSuccess;
}

int64_t Event::time(bool getStartTs) const {
  assert(event_ != nullptr);
  if (getStartTs) {
    return static_cast<int64_t>(event_->profilingInfo().start_);
  } else {
    return static_cast<int64_t>(event_->profilingInfo().end_);
  }
}

int64_t EventDD::time(bool getStartTs) const {
  uint64_t start = 0, end = 0;
  assert(event_ != nullptr);
  g_devices[deviceId()]->devices()[0]->getHwEventTime(*event_, &start, &end);
  // FIXME: This is only needed if the command had to wait CL_COMPLETE status
  if (start == 0 || end == 0) {
    return Event::time(getStartTs);
  }
  if (getStartTs) {
    return static_cast<int64_t>(start);
  } else {
    return static_cast<int64_t>(end);
  }
}

hipError_t Event::streamWaitCommand(amd::Command*& command, hip::Stream* stream) {
  amd::Command::EventWaitList eventWaitList;
  if (event_ != nullptr) {
    eventWaitList.push_back(event_);
  }
  command = new amd::Marker(*stream, kMarkerDisableFlush, eventWaitList);
  // Since we only need to have a dependency on an existing event,
  // we may not need to flush any caches.
  command->setEventScope(amd::Device::kCacheStateIgnore);

  if (command == NULL) {
    return hipErrorOutOfMemory;
  }
  return hipSuccess;
}

hipError_t Event::enqueueStreamWaitCommand(hipStream_t stream, amd::Command* command) {
  command->enqueue();
  return hipSuccess;
}

hipError_t Event::streamWait(hipStream_t stream, uint flags) {
  hip::Stream* hip_stream = hip::getStream(stream);
  // Access to event_ object must be lock protected
  amd::ScopedLock lock(lock_);
  if ((event_ == nullptr) || (event_->command().queue() == hip_stream) || ready()) {
    return hipSuccess;
  }
  if (!event_->notifyCmdQueue()) {
    return hipErrorLaunchOutOfResources;
  }
  amd::Command* command;
  hipError_t status = streamWaitCommand(command, hip_stream);
  if (status != hipSuccess) {
    return status;
  }
  status = enqueueStreamWaitCommand(stream, command);
  if (status != hipSuccess) {
    return status;
  }
  command->release();
  return hipSuccess;
}

// ================================================================================================
hipError_t Event::recordCommand(amd::Command*& command, amd::HostQueue* stream,
                                uint32_t ext_flags, bool batch_flush) {
  if (command == nullptr) {
    int32_t releaseFlags = ((ext_flags == 0) ? flags_ : ext_flags) &
                            (hipEventReleaseToDevice | hipEventReleaseToSystem |
                             hipEventDisableSystemFence);
    if (releaseFlags & hipEventDisableSystemFence) {
      releaseFlags = amd::Device::kCacheStateIgnore;
    } else {
      releaseFlags = amd::Device::kCacheStateInvalid;
    }
    // Always submit a EventMarker.
    command = new hip::EventMarker(*stream, !kMarkerDisableFlush, true, releaseFlags, batch_flush);
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t Event::enqueueRecordCommand(hipStream_t stream, amd::Command* command, bool record) {
  command->enqueue();
  if (event_ == &command->event()) return hipSuccess;
  if (event_ != nullptr) {
    event_->release();
  }
  event_ = &command->event();
  unrecorded_ = !record;

  return hipSuccess;
}

// ================================================================================================
hipError_t Event::addMarker(hipStream_t stream, amd::Command* command,
                            bool record, bool batch_flush) {
  hip::Stream* hip_stream = hip::getStream(stream);
  // Keep the lock always at the beginning of this to avoid a race. SWDEV-277847
  amd::ScopedLock lock(lock_);
  hipError_t status = recordCommand(command, hip_stream, 0, batch_flush);
  if (status != hipSuccess) {
    return hipSuccess;
  }
  status = enqueueRecordCommand(stream, command, record);
  return status;
}

// ================================================================================================
bool isValid(hipEvent_t event) {
  // NULL event is always valid
  if (event == nullptr) {
    return true;
  }

  amd::ScopedLock lock(eventSetLock);
  if (eventSet.find(event) == eventSet.end()) {
    return false;
  }

  return true;
}

// ================================================================================================
hipError_t ihipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  unsigned supportedFlags = hipEventDefault | hipEventBlockingSync | hipEventDisableTiming |
                            hipEventReleaseToDevice | hipEventReleaseToSystem |
                            hipEventInterprocess | hipEventDisableSystemFence;

  const unsigned releaseFlags = (hipEventReleaseToDevice | hipEventReleaseToSystem |
                                 hipEventDisableSystemFence);
  // can't set any unsupported flags.
  // can set only one of the release flags.
  // if hipEventInterprocess flag is set, then hipEventDisableTiming flag also must be set
  const bool illegalFlags = (flags & ~supportedFlags) ||
                            ([](unsigned int num){
                              unsigned int bitcount;
                              for (bitcount = 0; num; bitcount++) {
                                num &= num - 1;
                              }
                              return bitcount; } (flags & releaseFlags) > 1) ||
                            ((flags & hipEventInterprocess) && !(flags & hipEventDisableTiming));
  if (!illegalFlags) {
    hip::Event* e = nullptr;
    if (flags & hipEventInterprocess) {
      e = new hip::IPCEvent();
    } else {
      if (AMD_DIRECT_DISPATCH) {
        e = new hip::EventDD(flags);
      } else {
        e = new hip::Event(flags);
      }
    }
    // App might have used combination of flags i.e. hipEventInterprocess|hipEventDisableTiming
    // However based on hipEventInterprocess flag, IPCEvent creates even with
    // JUST hipEventInterprocess and hence, Actual hipEventInterprocess|hipEventDisableTiming
    // flag is getting supressed with hipEventInterprocess
    e->flags_ = flags;
    if (e == nullptr) {
      return hipErrorOutOfMemory;
    }
    *event = reinterpret_cast<hipEvent_t>(e);
    amd::ScopedLock lock(hip::eventSetLock);
    hip::eventSet.insert(*event);
  } else {
    return hipErrorInvalidValue;
  }
  return hipSuccess;
}

hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned flags) {
  HIP_INIT_API(hipEventCreateWithFlags, event, flags);

  if (event == nullptr) {
    return hipErrorInvalidValue;
  }

  HIP_RETURN(ihipEventCreateWithFlags(event, flags), *event);
}

hipError_t hipEventCreate(hipEvent_t* event) {
  HIP_INIT_API(hipEventCreate, event);

  if (event == nullptr) {
    return hipErrorInvalidValue;
  }

  HIP_RETURN(ihipEventCreateWithFlags(event, 0), *event);
}

hipError_t hipEventDestroy(hipEvent_t event) {
  HIP_INIT_API(hipEventDestroy, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  amd::ScopedLock lock(hip::eventSetLock);
  if (hip::eventSet.erase(event) == 0 ) {
    return hipErrorContextIsDestroyed;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  // There is a possibility that stream destroy be called first
  hipStream_t s = e->GetCaptureStream();
  if (hip::isValid(s)) {
    if (e->GetCaptureStream() != nullptr) {
      reinterpret_cast<hip::Stream*>(e->GetCaptureStream())->EraseCaptureEvent(event);
    }
  }
  delete e;
  HIP_RETURN(hipSuccess);
}

hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop) {
  HIP_INIT_API(hipEventElapsedTime, ms, start, stop);

  if (ms == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  if (start == nullptr || stop == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  hip::Event* eStart = reinterpret_cast<hip::Event*>(start);
  hip::Event* eStop = reinterpret_cast<hip::Event*>(stop);

  if (eStart->deviceId() != eStop->deviceId()) {
    HIP_RETURN(hipErrorInvalidHandle);
  }

  HIP_RETURN(eStart->elapsedTime(*eStop, *ms), "Elapsed Time = ", *ms);
}

hipError_t hipEventRecord_common(hipEvent_t event, hipStream_t stream) {
  hipError_t status = hipSuccess;
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }
  getStreamPerThread(stream);
  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(stream);
  hip::Stream* hip_stream = hip::getStream(stream);
  e->SetCaptureStream(stream);
  if ((s != nullptr) && (s->GetCaptureStatus() == hipStreamCaptureStatusActive)) {
    ClPrint(amd::LOG_INFO, amd::LOG_API,
        "[hipGraph] Current capture node EventRecord on stream : %p, Event %p", stream, event);
    s->SetCaptureEvent(event);
    std::vector<hip::GraphNode*> lastCapturedNodes = s->GetLastCapturedNodes();
    if (!lastCapturedNodes.empty()) {
      e->SetNodesPrevToRecorded(lastCapturedNodes);
    }
  } else {
    if (g_devices[e->deviceId()]->devices()[0] != &hip_stream->device()) {
      return hipErrorInvalidHandle;
    }
    status = e->addMarker(stream, nullptr, true, !hip::Event::kBatchFlush);
  }
  return status;
}

hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(hipEventRecord, event, stream);
  HIP_RETURN(hipEventRecord_common(event, stream));
}

hipError_t hipEventRecord_spt(hipEvent_t event, hipStream_t stream) {
  HIP_INIT_API(hipEventRecord, event, stream);
  PER_THREAD_DEFAULT_STREAM(stream);
  HIP_RETURN(hipEventRecord_common(event, stream));
}

hipError_t hipEventSynchronize(hipEvent_t event) {
  HIP_INIT_API(hipEventSynchronize, event);

  if (event == nullptr) {
    HIP_RETURN(hipErrorInvalidHandle);
  }
  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(e->GetCaptureStream());
  if ((s != nullptr) && (s->GetCaptureStatus() == hipStreamCaptureStatusActive)) {
      s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
      HIP_RETURN(hipErrorCapturedEvent);
  }
  if (hip::Stream::StreamCaptureOngoing(e->GetCaptureStream()) == true) {
    HIP_RETURN(hipErrorStreamCaptureUnsupported);
  }

  hipError_t status = e->synchronize();
  // Release freed memory for all memory pools on the device
  g_devices[e->deviceId()]->ReleaseFreedMemory();
  HIP_RETURN(status);
}

hipError_t ihipEventQuery(hipEvent_t event) {
  if (event == nullptr) {
    return hipErrorInvalidHandle;
  }

  hip::Event* e = reinterpret_cast<hip::Event*>(event);
  hip::Stream* s = reinterpret_cast<hip::Stream*>(e->GetCaptureStream());
  if ((s != nullptr) && (s->GetCaptureStatus() == hipStreamCaptureStatusActive)) {
    s->SetCaptureStatus(hipStreamCaptureStatusInvalidated);
    HIP_RETURN(hipErrorCapturedEvent);
  }
  return e->query();
}

hipError_t hipEventQuery(hipEvent_t event) {
  HIP_INIT_API(hipEventQuery, event);
  HIP_RETURN(ihipEventQuery(event));
}
}  // namespace hip
