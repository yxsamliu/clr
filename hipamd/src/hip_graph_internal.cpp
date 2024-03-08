/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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

#include "hip_graph_internal.hpp"
#include <queue>

#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    case_string = #C;                                                                              \
    break;
const char* GetGraphNodeTypeString(uint32_t op) {
  const char* case_string;
  switch (static_cast<hipGraphNodeType>(op)) {
    CASE_STRING(hipGraphNodeTypeKernel, KernelNode)
    CASE_STRING(hipGraphNodeTypeMemcpy, MemcpyNode)
    CASE_STRING(hipGraphNodeTypeMemset, MemsetNode)
    CASE_STRING(hipGraphNodeTypeHost, HostNode)
    CASE_STRING(hipGraphNodeTypeGraph, GraphNode)
    CASE_STRING(hipGraphNodeTypeEmpty, EmptyNode)
    CASE_STRING(hipGraphNodeTypeWaitEvent, WaitEventNode)
    CASE_STRING(hipGraphNodeTypeEventRecord, EventRecordNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreSignal, ExtSemaphoreSignalNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreWait, ExtSemaphoreWaitNode)
    CASE_STRING(hipGraphNodeTypeMemAlloc, MemAllocNode)
    CASE_STRING(hipGraphNodeTypeMemFree, MemFreeNode)
    CASE_STRING(hipGraphNodeTypeMemcpyFromSymbol, MemcpyFromSymbolNode)
    CASE_STRING(hipGraphNodeTypeMemcpyToSymbol, MemcpyToSymbolNode)
    default:
      case_string = "Unknown node type";
  };
  return case_string;
};

namespace hip {
int GraphNode::nextID = 0;
int Graph::nextID = 0;
std::unordered_set<GraphNode*> GraphNode::nodeSet_;
amd::Monitor GraphNode::nodeSetLock_{"Guards global node set"};
std::unordered_set<Graph*> Graph::graphSet_;
amd::Monitor Graph::graphSetLock_{"Guards global graph set"};
std::unordered_set<GraphExec*> GraphExec::graphExecSet_;
amd::Monitor GraphExec::graphExecSetLock_{"Guards global exec graph set"};
std::unordered_set<UserObject*> UserObject::ObjectSet_;
amd::Monitor UserObject::UserObjectLock_{"Guards global user object"};

hipError_t GraphMemcpyNode1D::ValidateParams(void* dst, const void* src, size_t count,
                                                hipMemcpyKind kind) {
  hipError_t status = ihipMemcpy_validate(dst, src, count, kind);
  if (status != hipSuccess) {
    return status;
  }
  size_t sOffsetOrig = 0;
  amd::Memory* origSrcMemory = getMemoryObject(src, sOffsetOrig);
  size_t dOffsetOrig = 0;
  amd::Memory* origDstMemory = getMemoryObject(dst, dOffsetOrig);

  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObject(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObject(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
    if (origDstMemory->getContext().devices()[0] != dstMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if ((kind != hipMemcpyHostToDevice) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
    if (origSrcMemory->getContext().devices()[0] != srcMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if ((kind != hipMemcpyDeviceToHost) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory != nullptr)) {
    if (origDstMemory->getContext().devices()[0] != dstMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
    if (origSrcMemory->getContext().devices()[0] != srcMemory->getContext().devices()[0]) {
      return hipErrorInvalidValue;
    }
  }
  return hipSuccess;
}

hipError_t GraphMemcpyNode::ValidateParams(const hipMemcpy3DParms* pNodeParams) {
  hipError_t status;
  status = ihipMemcpy3D_validate(pNodeParams);
  if (status != hipSuccess) {
    return status;
  }

  const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pNodeParams);
  status = ihipDrvMemcpy3D_validate(&pCopy);
  if (status != hipSuccess) {
    return status;
  }
  return hipSuccess;
}

bool Graph::isGraphValid(Graph* pGraph) {
  amd::ScopedLock lock(graphSetLock_);
  if (graphSet_.find(pGraph) == graphSet_.end()) {
    return false;
  }
  return true;
}

void Graph::AddNode(const Node& node) {
  vertices_.emplace_back(node);
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Add %s(%p)",
          GetGraphNodeTypeString(node->GetType()), node);
  node->SetParentGraph(this);
}

void Graph::RemoveNode(const Node& node) {
  vertices_.erase(std::remove(vertices_.begin(), vertices_.end(), node), vertices_.end());
  delete node;
}

// root nodes are all vertices with 0 in-degrees
std::vector<Node> Graph::GetRootNodes() const {
  std::vector<Node> roots;
  for (auto entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      roots.push_back(entry);
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Root node: %s(%p)",
              GetGraphNodeTypeString(entry->GetType()), entry);
    }
  }
  return roots;
}

// leaf nodes are all vertices with 0 out-degrees
std::vector<Node> Graph::GetLeafNodes() const {
  std::vector<Node> leafNodes;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      leafNodes.push_back(entry);
    }
  }
  return leafNodes;
}

size_t Graph::GetLeafNodeCount() const {
  int numLeafNodes = 0;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      numLeafNodes++;
    }
  }
  return numLeafNodes;
}

std::vector<std::pair<Node, Node>> Graph::GetEdges() const {
  std::vector<std::pair<Node, Node>> edges;
  for (const auto& i : vertices_) {
    for (const auto& j : i->GetEdges()) {
      edges.push_back(std::make_pair(i, j));
    }
  }
  return edges;
}

void Graph::GetRunListUtil(Node v, std::unordered_map<Node, bool>& visited,
                               std::vector<Node>& singleList,
                               std::vector<std::vector<Node>>& parallelLists,
                               std::unordered_map<Node, std::vector<Node>>& dependencies) {
  // Mark the current node as visited.
  visited[v] = true;
  singleList.push_back(v);
  // Recurse for all the vertices adjacent to this vertex
  for (auto& adjNode : v->GetEdges()) {
    if (!visited[adjNode]) {
      // For the parallel list nodes add parent as the dependency
      if (singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] For %s(%p) - add parent as dependency %s(%p)",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
      GetRunListUtil(adjNode, visited, singleList, parallelLists, dependencies);
    } else {
      for (auto& list : parallelLists) {
        // Merge singleList when adjNode matches with the first element of the list in existing
        // lists
        if (adjNode == list[0]) {
          for (auto k = singleList.rbegin(); k != singleList.rend(); ++k) {
            list.insert(list.begin(), *k);
          }
          singleList.erase(singleList.begin(), singleList.end());
        }
      }
      // If the list cannot be merged with the existing list add as dependancy
      if (!singleList.empty()) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] For %s(%p) - add dependency %s(%p)",
                GetGraphNodeTypeString(adjNode->GetType()), adjNode,
                GetGraphNodeTypeString(v->GetType()), v);
        dependencies[adjNode].push_back(v);
      }
    }
  }
  if (!singleList.empty()) {
    parallelLists.push_back(singleList);
    singleList.erase(singleList.begin(), singleList.end());
  }
}
// The function to do Topological Sort.
// It uses recursive GetRunListUtil()
void Graph::GetRunList(std::vector<std::vector<Node>>& parallelLists,
                           std::unordered_map<Node, std::vector<Node>>& dependencies) {
  std::vector<Node> singleList;

  // Mark all the vertices as not visited
  std::unordered_map<Node, bool> visited;
  for (auto node : vertices_) visited[node] = false;

  // Call the recursive helper function for all vertices one by one
  for (auto node : vertices_) {
    // If the node has embedded child graph
    node->GetRunList(parallelLists, dependencies);
    if (visited[node] == false) {
      GetRunListUtil(node, visited, singleList, parallelLists, dependencies);
    }
  }
  for (size_t i = 0; i < parallelLists.size(); i++) {
    for (size_t j = 0; j < parallelLists[i].size(); j++) {
      ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] List %d - %s(%p)", i + 1,
              GetGraphNodeTypeString(parallelLists[i][j]->GetType()), parallelLists[i][j]);
    }
  }
}
bool Graph::TopologicalOrder(std::vector<Node>& TopoOrder) {
  std::queue<Node> q;
  std::unordered_map<Node, int> inDegree;
  for (auto entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      q.push(entry);
    }
    inDegree[entry] = entry->GetInDegree();
  }
  while (!q.empty())
  {
    Node node = q.front();
    TopoOrder.push_back(node);
    q.pop();
    for (auto edge : node->GetEdges()) {
      inDegree[edge]--;
      if (inDegree[edge] == 0) {
        q.push(edge);
      }
    }
  }
  if (GetNodeCount() == TopoOrder.size()) {
    return true;
  }
  return false;
}
Graph* Graph::clone(std::unordered_map<Node, Node>& clonedNodes) const {
  Graph* newGraph = new Graph(device_, this);
  for (auto entry : vertices_) {
    GraphNode* node = entry->clone();
    node->SetParentGraph(newGraph);
    newGraph->vertices_.push_back(node);
    clonedNodes[entry] = node;
  }

  std::vector<Node> clonedEdges;
  std::vector<Node> clonedDependencies;
  for (auto node : vertices_) {
    const std::vector<Node>& edges = node->GetEdges();
    clonedEdges.clear();
    for (auto edge : edges) {
      clonedEdges.push_back(clonedNodes[edge]);
    }
    clonedNodes[node]->SetEdges(clonedEdges);
  }
  for (auto node : vertices_) {
    const std::vector<Node>& dependencies = node->GetDependencies();
    clonedDependencies.clear();
    for (auto dep : dependencies) {
      clonedDependencies.push_back(clonedNodes[dep]);
    }
    clonedNodes[node]->SetDependencies(clonedDependencies);
  }
  return newGraph;
}

Graph* Graph::clone() const {
  std::unordered_map<Node, Node> clonedNodes;
  return clone(clonedNodes);
}

bool GraphExec::isGraphExecValid(GraphExec* pGraphExec) {
  amd::ScopedLock lock(graphExecSetLock_);
  if (graphExecSet_.find(pGraphExec) == graphExecSet_.end()) {
    return false;
  }
  return true;
}

hipError_t GraphExec::CreateStreams(uint32_t num_streams) {
  parallel_streams_.reserve(num_streams);
  for (uint32_t i = 0; i < num_streams; ++i) {
    auto stream = new hip::Stream(hip::getCurrentDevice(),
                                  hip::Stream::Priority::Normal, hipStreamNonBlocking);
    if (stream == nullptr || !stream->Create()) {
      if (stream != nullptr) {
        hip::Stream::Destroy(stream);
      }
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create parallel stream!");
      return hipErrorOutOfMemory;
    }
    parallel_streams_.push_back(stream);
  }
  return hipSuccess;
}

hipError_t GraphExec::Init() {
  hipError_t status = hipSuccess;
  size_t min_num_streams = 1;

  for (auto& node : topoOrder_) {
    status = node->GetNumParallelStreams(min_num_streams);
    if(status != hipSuccess) {
      return status;
    }
  }
  status = CreateStreams(parallelLists_.size() - 1 + min_num_streams);
  return status;
}

hipError_t GraphExec::CaptureAQLPackets() {
  hipError_t status = hipSuccess;
  if (parallelLists_.size() == 1) {
    size_t kernArgSizeForGraph = 0;
    hip::Stream* stream = nullptr;
    // GPU packet capture is enabled for kernel nodes. Calculate the kernel
    // arg size required for all graph kernel nodes to allocate
    for (const auto& list : parallelLists_) {
      stream = GetAvailableStreams();
      for (auto& node : list) {
        node->SetStream(stream, this);
        if (node->GetType() == hipGraphNodeTypeKernel) {
          kernArgSizeForGraph += reinterpret_cast<hip::GraphKernelNode*>(node)->GetKerArgSize();
        }
      }
    }

    auto device = g_devices[ihipGetDevice()]->devices()[0];
    if (device->info().largeBar_) {
      kernarg_pool_graph_ =
          reinterpret_cast<address>(device->deviceLocalAlloc(kernArgSizeForGraph));
      device_kernarg_pool_ = true;
    } else {
      kernarg_pool_graph_ = reinterpret_cast<address>(
          device->hostAlloc(kernArgSizeForGraph, 0, amd::Device::MemorySegment::kKernArg));
    }

    if (kernarg_pool_graph_ == nullptr) {
      return hipErrorMemoryAllocation;
    }
    kernarg_pool_size_graph_ = kernArgSizeForGraph;

    for (auto& node : topoOrder_) {
      if (node->GetType() == hipGraphNodeTypeKernel) {
        auto kernelNode = reinterpret_cast<hip::GraphKernelNode*>(node);
        status = node->CreateCommand(node->GetQueue());
        // From the kernel pool allocate the kern arg size required for the current kernel node.
        address kernArgOffset = allocKernArg(kernelNode->GetKernargSegmentByteSize(),
                                             kernelNode->GetKernargSegmentAlignment());
        if (kernArgOffset == nullptr) {
          return hipErrorMemoryAllocation;
        }
        // Form GPU packet capture for the kernel node.
        kernelNode->CaptureAndFormPacket(kernArgOffset);
      }
    }

    if (device_kernarg_pool_ && !device->isXgmi()) {
      *device->info().hdpMemFlushCntl = 1u;
      if (*device->info().hdpMemFlushCntl != UINT32_MAX) {
        LogError("Unexpected HDP Register readback value!");
      }
    }

    ResetQueueIndex();
  }
  return status;
}

hipError_t FillCommands(std::vector<std::vector<Node>>& parallelLists,
                        std::unordered_map<Node, std::vector<Node>>& nodeWaitLists,
                        std::vector<Node>& topoOrder, Graph* clonedGraph,
                        amd::Command*& graphStart, amd::Command*& graphEnd, hip::Stream* stream) {
  hipError_t status = hipSuccess;
  for (auto& node : topoOrder) {
    // TODO: clone commands from next launch
    status = node->CreateCommand(node->GetQueue());
    if (status != hipSuccess) return status;
    amd::Command::EventWaitList waitList;
    for (auto depNode : nodeWaitLists[node]) {
      for (auto command : depNode->GetCommands()) {
        waitList.push_back(command);
      }
    }
    node->UpdateEventWaitLists(waitList);
  }

  std::vector<Node> rootNodes = clonedGraph->GetRootNodes();
  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] RootCommand get launched on stream %p", stream);

  for (auto& root : rootNodes) {
    //If rootnode is launched on to the same stream dont add dependency
    if (root->GetQueue() != stream) {
      if (graphStart == nullptr) {
        graphStart = new amd::Marker(*stream, false, {});
        if (graphStart == nullptr) {
          return hipErrorOutOfMemory;
        }
      }
      amd::Command::EventWaitList waitList;
      waitList.push_back(graphStart);
      auto commands = root->GetCommands();
      if (!commands.empty()) {
        commands[0]->updateEventWaitList(waitList);
      }
    }
  }

  // graphEnd ensures next enqueued ones start after graph is finished (all parallel branches)
  amd::Command::EventWaitList graphLastCmdWaitList;
  std::vector<Node> leafNodes = clonedGraph->GetLeafNodes();

  for (auto& leaf : leafNodes) {
    // If leaf node is launched on to the same stream dont add dependency
    if (leaf->GetQueue() != stream) {
      amd::Command::EventWaitList waitList;
      waitList.push_back(graphEnd);
      auto commands = leaf->GetCommands();
      if (!commands.empty()) {
        graphLastCmdWaitList.push_back(commands.back());
      }
    }
  }
  if (!graphLastCmdWaitList.empty()) {
    graphEnd = new amd::Marker(*stream, false, graphLastCmdWaitList);
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] EndCommand will get launched on stream %p", stream);
    if (graphEnd == nullptr) {
      graphStart->release();
      return hipErrorOutOfMemory;
    }
  }
  return hipSuccess;
}

void UpdateStream(std::vector<std::vector<Node>>& parallelLists, hip::Stream* stream,
                 GraphExec* ptr) {
  int i = 0;
  for (const auto& list : parallelLists) {
    // first parallel list will be launched on the same queue as parent
    if (i == 0) {
      for (auto& node : list) {
        node->SetStream(stream, ptr);
      }
    } else {  // New stream for parallel branches
      hip::Stream* stream = ptr->GetAvailableStreams();
      for (auto& node : list) {
        node->SetStream(stream, ptr);
      }
    }
    i++;
  }
}

hipError_t GraphExec::Run(hipStream_t stream) {
  hipError_t status = hipSuccess;

  if (hip::getStream(stream) == nullptr) {
    return hipErrorInvalidResourceHandle;
  }
  auto hip_stream = (stream == nullptr) ? hip::getCurrentDevice()->NullStream()
                                        : reinterpret_cast<hip::Stream*>(stream);
  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (!topoOrder_.empty()) {
      topoOrder_[0]->GetParentGraph()->FreeAllMemory(hip_stream);
    }
  }

  // If this is a repeat launch, make sure corresponding MemFreeNode exists for a MemAlloc node
  if (repeatLaunch_ == true) {
    for (auto& node : topoOrder_) {
      if (node->GetType() == hipGraphNodeTypeMemAlloc &&
          static_cast<GraphMemAllocNode*>(node)->IsActiveMem() == true) {
        return hipErrorInvalidValue;
      }
    }
  } else {
    repeatLaunch_ = true;
  }

  if (parallelLists_.size() == 1) {
    amd::AccumulateCommand* accumulate = nullptr;
    bool isLastPacketKernel = false;
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE) {
      uint8_t* lastCapturedPacket = (topoOrder_.back()->GetType() == hipGraphNodeTypeKernel) ?
                                  topoOrder_.back()->GetAqlPacket() : nullptr;
      accumulate = new amd::AccumulateCommand(*hip_stream, {}, nullptr, lastCapturedPacket);
    }

    for (int i = 0; i < topoOrder_.size() - 1; i++) {
      if (DEBUG_CLR_GRAPH_PACKET_CAPTURE && topoOrder_[i]->GetType() == hipGraphNodeTypeKernel) {
        if (topoOrder_[i]->GetEnabled()) {
          hip_stream->vdev()->dispatchAqlPacket(topoOrder_[i]->GetAqlPacket(), accumulate);
          accumulate->addKernelName(topoOrder_[i]->GetKernelName());
        }
      } else {
        topoOrder_[i]->SetStream(hip_stream, this);
        status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
        topoOrder_[i]->EnqueueCommands(stream);
      }
    }

    // If last captured packet is kernel, optimize to detect completion of last kernel
    // This saves on extra packet submitted to determine end of graph
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE && topoOrder_.back()->GetType() == hipGraphNodeTypeKernel) {
      // Add the last kernel node name to the accumulate command
      accumulate->addKernelName(topoOrder_.back()->GetKernelName());
      accumulate->enqueue();
      accumulate->release();
      isLastPacketKernel = true;
    } else {
      topoOrder_.back()->SetStream(hip_stream, this);
      status = topoOrder_.back()->CreateCommand(topoOrder_.back()->GetQueue());
      topoOrder_.back()->EnqueueCommands(stream);
    }

    // If last packet is not kernel, submit a marker to detect end of graph
    if (DEBUG_CLR_GRAPH_PACKET_CAPTURE && !isLastPacketKernel) {
      accumulate->enqueue();
      accumulate->release();
    }
  } else {
    UpdateStream(parallelLists_, hip_stream, this);
    amd::Command* rootCommand = nullptr;
    amd::Command* endCommand = nullptr;
    status = FillCommands(parallelLists_, nodeWaitLists_, topoOrder_, clonedGraph_, rootCommand,
                          endCommand, hip_stream);
    if (status != hipSuccess) {
      return status;
    }
    if (rootCommand != nullptr) {
      rootCommand->enqueue();
      rootCommand->release();
    }
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->EnqueueCommands(stream);
    }
    if (endCommand != nullptr) {
      endCommand->enqueue();
      endCommand->release();
    }
  }
  ResetQueueIndex();
  return status;
}
}  // namespace hip
