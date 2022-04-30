#ifndef EXPERIMENT_SERVER_H_
#define EXPERIMENT_SERVER_H_

#include <thrust/system/cuda/experimental/pinned_allocator.h>

#include "folly/MPMCQueue.h"
#include "omp.h"

#include "data_loader.h"
#include "inference_client.h"
#include "ipc.h"
#include "common.h"

class ExperimentServer {
 private:
  const DataLoader& kLoader_;
  InferenceClient *kInfer_; // not const cause this would be a pain
  const size_t kBatchSize_;
  const size_t kImSize_;

  const bool kRunInfer_;

 public:
  ExperimentServer(
      const DataLoader& kLoader, InferenceClient *kInfer,
      const size_t kBatchSize, const bool kRunInfer) :
      kLoader_(kLoader), kInfer_(kInfer), kBatchSize_(kBatchSize),
      kImSize_(3 * kLoader.GetResol() * kLoader.GetResol()),
      kRunInfer_(kRunInfer) {
    // for (size_t i = 0; i < omp_get_max_threads() * 3; i++)
    //   batch_queue_.blockingWrite(
    //       std::make_unique<std::vector<float, thrust::system::cuda::experimental::pinned_allocator<float> > >(
    //           kBatchSize_ * kImSize_));
  }

  void RunInferenceOnFiles(const std::vector<std::string>& kFileNames, std::vector<float> &output);
  float TimeEndToEnd(const std::vector<std::string>& kFileNames, std::vector<float> &output);

  void RunInferenceOnCompressed(const std::vector<CompressedImage>& kCompressedImages, std::vector<float> &output);
  float TimeNoLoad(const std::vector<CompressedImage>& kCompressedImages, std::vector<float> &output);

  float TimeInferenceOnly();

  float TimeDecodePreprocOnly(const std::vector<CompressedImage>& kCompressedImages);
};

#endif // EXPERIMENT_SERVER_H_
