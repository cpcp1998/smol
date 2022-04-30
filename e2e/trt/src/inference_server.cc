#include <assert.h>
#include <future>
#include <numeric>
#include <thread>
#include <vector>

#include "omp.h"

#include "calibrator.h"
#include "inference_server.h"


void InferenceServer::_RunInferenceThread(const size_t idx) {
  std::vector<char> buffer;
  while (true) {
    size_t isize, osize;
    void *output;
    void *addr = inputQueue.startRead(&isize);
    if (!addr) return;
    if (isize > buffer.size()) buffer.resize(isize);
    memcpy(buffer.data(), addr, isize);
    inputQueue.finishRead(addr);

    RunInference(buffer.data(), isize, output, osize, idx);

    if (!output || !osize) continue;
    addr = outputQueue.alloc(osize);
    if(!addr) continue;
    memcpy(addr, output, osize);
    outputQueue.finishWrite(addr);
  }
}

static void add_resize(nvinfer1::INetworkDefinition *network, const int32_t kBatchSize) {
  using namespace nvinfer1;
  // FIXME: 141, 224
  nvinfer1::ITensor* old_input = network->getInput(0);
  nvinfer1::ILayer* first_layer = network->getLayer(0);

  nvinfer1::ITensor* new_input = network->addInput("thumbnail_im", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{kBatchSize, 3, 141, 141});
  nvinfer1::IResizeLayer* resizeLayer = network->addResize(*new_input);
  resizeLayer->setOutputDimensions(nvinfer1::Dims4(kBatchSize, 3, 224, 224));
  resizeLayer->setResizeMode(nvinfer1::ResizeMode::kLINEAR);

  nvinfer1::ITensor* resize_output = resizeLayer->getOutput(0);
  first_layer->setInput(0, *resize_output);
  network->removeTensor(*old_input);
}

nvinfer1::ICudaEngine* OnnxInferenceServer::CreateCudaEngine(
    const std::string& kOnnxPath,
    const std::string& kOnnxPathBS1,
    BaseCalibrator *calibrator,
    const bool kDoINT8,
    const bool kAddResize) {
  using namespace std;
  using namespace nvinfer1;
  using nvonnxparser::IParser;
  unique_ptr<IBuilder> builder{createInferBuilder(gLogger)};
  builder->setMaxBatchSize(kBatchSize_);
  unique_ptr<IBuilderConfig> config{builder->createBuilderConfig()};
  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  unique_ptr<INetworkDefinition> network{builder->createNetworkV2(explicitBatch)};
  unique_ptr<IParser> parser{nvonnxparser::createParser(*network, gLogger)};

  if (!parser->parseFromFile(kOnnxPath.c_str(), static_cast<int>(ILogger::Severity::kINFO))) {
    throw "ERROR: could not parse input engine.";
  }

  // FIXME: is this always correct?
  const int32_t kBatchSize = network->getInput(0)->getDimensions().d[0];

  if (kAddResize) {
    add_resize(network.get(), kBatchSize);
  }

  config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
  if (kDoINT8) {
    std::cerr << "Doing INT8" << std::endl;

    config->setFlag(BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator);
  } else {
    config->setFlag(BuilderFlag::kFP16);
  }

  unique_ptr<IHostMemory> serializedModel{builder->buildSerializedNetwork(*network, *config)};
  unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
  return runtime->deserializeCudaEngine(serializedModel->data(), serializedModel->size());
}

nvinfer1::ICudaEngine* OnnxInferenceServer::GetCudaEngine(const std::string& kEnginePath) {
  using namespace std;
  using namespace nvinfer1;
  ICudaEngine* engine{nullptr};

  string buffer = readBuffer(kEnginePath);
  if (buffer.size()) {
    // Try to deserialize engine.
    unique_ptr<IRuntime> runtime{createInferRuntime(gLogger)};
    engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size());
  }

  if (!engine) {
    throw std::runtime_error("Failed to load engine");
  }
  return engine;
}

OnnxInferenceServer::OnnxInferenceServer(
    const std::string &inputQueueName,
    const std::string &outputQueueName,
    const std::string& kEnginePath, const size_t kBatchSize, const bool kDoMemcpy) :
    InferenceServer(inputQueueName, outputQueueName),
    kBatchSize_(kBatchSize),
    contexts(kNbStreams_), outputBuffers(kNbStreams_),
    kDoMemcpy_(kDoMemcpy) {
  // TensorRT engine stuff
  this->engine.reset(GetCudaEngine(kEnginePath));

  LoadAndLaunch();
}

OnnxInferenceServer::OnnxInferenceServer(
    const std::string &inputQueueName,
    const std::string &outputQueueName,
    const std::string& kOnnxPath, const std::string& kOnnxPathBS1,
    const std::string& kCachePath,
    const size_t kBatchSize, const bool kDoMemcpy,
    const DataLoader *kLoader,
    const std::vector<CompressedImage>& kCompressedImages,
    const bool kDoINT8,
    const bool kAddResize
) :
    InferenceServer(inputQueueName, outputQueueName),
    kBatchSize_(kBatchSize),
    contexts(kNbStreams_), outputBuffers(kNbStreams_),
    kDoMemcpy_(kDoMemcpy) {
  BaseCalibrator *calibrator = NULL;
  if (kDoINT8) {
    assert(kLoader != nullptr);
    calibrator = new ImageCalibrator(kLoader, kCompressedImages, kBatchSize);
  }
  this->engine.reset(
      CreateCudaEngine(kOnnxPath, kOnnxPathBS1,
                       calibrator,
                       kDoINT8, kAddResize));
  // Cache engine
  std::unique_ptr<nvinfer1::IHostMemory> engine_plan{engine->serialize()};
  nvinfer1::writeBuffer(engine_plan->data(), engine_plan->size(), kCachePath);

  LoadAndLaunch();
}


OnnxInferenceServer::OnnxInferenceServer(
    const std::string &inputQueueName,
    const std::string &outputQueueName,
    const std::string& kOnnxPath, const std::string& kOnnxPathBS1,
    const std::string& kCachePath,
    const size_t kBatchSize, const bool kDoMemcpy,
    const VideoDataLoader *kLoader,
    const std::vector<std::string>& kFileNames,
    const bool kDoINT8,
    const bool kAddResize
) :
    InferenceServer(inputQueueName, outputQueueName),
    kBatchSize_(kBatchSize),
    contexts(kNbStreams_), outputBuffers(kNbStreams_),
    kDoMemcpy_(kDoMemcpy) {
  BaseCalibrator *calibrator = NULL;
  if (kDoINT8) {
    assert(kLoader != nullptr);
    calibrator = new VideoCalibrator(kLoader, kFileNames, kBatchSize);
  }
  this->engine.reset(
      CreateCudaEngine(kOnnxPath, kOnnxPathBS1,
                       calibrator,
                       kDoINT8, kAddResize));
  // Cache engine
  std::unique_ptr<nvinfer1::IHostMemory> engine_plan{engine->serialize()};
  nvinfer1::writeBuffer(engine_plan->data(), engine_plan->size(), kCachePath);

  LoadAndLaunch();
}


void OnnxInferenceServer::LoadAndLaunch() {
  // Currently, we only support exactly one input/output tensor
  assert(this->engine->getNbBindings() == 2);
  assert(this->engine->bindingIsInput(0) ^ this->engine->bindingIsInput(1));

  for (size_t j = 0; j < kNbStreams_; j++) {
    for (size_t i = 0; i < this->engine->getNbBindings(); i++) {
      nvinfer1::Dims dims{this->engine->getBindingDimensions(i)};
      size_t size = std::accumulate(dims.d, dims.d + dims.nbDims, 1, std::multiplies<size_t>());
      cudaMalloc(&this->bindings[j][i], size * sizeof(float) * kBatchSize_);
      if (!this->engine->bindingIsInput(i))
        kOutputSingle_ = size;
    }
    this->contexts[j].reset(engine->createExecutionContext());
    this->outputBuffers[j].resize(kOutputSingle_ * kBatchSize_ * sizeof(float));
  }

  // Launch thread
  for (size_t i = 0; i < kNbStreams_; i++)
    threads_.push_back(std::thread([this, i]{ _RunInferenceThread(i); }));
}



void OnnxInferenceServer::RunInference(void *data, size_t isize, void *&output, size_t &osize, size_t idx) {
  const int input_id = !contexts[idx]->getEngine().bindingIsInput(0);
  if (kDoMemcpy_) {
    cudaMemcpyAsync(bindings[idx][input_id],
                    data,
                    isize,
                    cudaMemcpyHostToDevice, streams[idx]);
  }
  contexts[idx]->enqueueV2(bindings[idx], streams[idx], nullptr);
  if (kDoMemcpy_) {
    cudaMemcpyAsync(outputBuffers[idx].data(),
                    bindings[idx][1 - input_id],
                    outputBuffers[idx].size(),
                    cudaMemcpyDeviceToHost, streams[idx]);
  }
  cudaStreamSynchronize(streams[idx]);
  output = outputBuffers[idx].data();
  osize = outputBuffers[idx].size();
}

void OnnxInferenceServer::Sync() {
}
