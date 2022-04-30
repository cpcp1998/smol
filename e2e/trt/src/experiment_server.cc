#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "experiment_server.h"

void ExperimentServer::RunInferenceOnFiles(
    const std::vector<std::string>& kFileNames,
    std::vector<float> &output) {
  std::vector<float> batch;
  batch.reserve(kBatchSize_ * kImSize_);

  #pragma omp parallel for
  for (size_t i = 0; i < kFileNames.size(); i += kBatchSize_) {
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kFileNames.size()) {
        kLoader_.LoadAndPreproc(
            kFileNames[i + j],
            batch.data() + j * kImSize_);
      }
    }
    size_t count = std::min(kFileNames.size() - i, kBatchSize_);
    kInfer_->RunInference(batch.data(), count * kImSize_ * sizeof(float));
  }
  kInfer_->sync();
}

float ExperimentServer::TimeEndToEnd(
    const std::vector<std::string>& kFileNames,
    std::vector<float> &output) {
  auto start = std::chrono::high_resolution_clock::now();
  RunInferenceOnFiles(kFileNames, output);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return diff.count() / 1000.0;
}

void ExperimentServer::RunInferenceOnCompressed(
    const std::vector<CompressedImage>& kCompressedImages,
    std::vector<float> &output) {
  #pragma omp parallel for
  for (size_t i = 0; i < kCompressedImages.size(); i += kBatchSize_) {
    std::vector<float> batch;
    batch.reserve(kBatchSize_ * kImSize_);
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kCompressedImages.size()) {
        kLoader_.DecodeAndPreproc(
            kCompressedImages[i + j],
            batch.data() + j * kImSize_);
      }
    }
    size_t count = std::min(kCompressedImages.size() - i, kBatchSize_);
    kInfer_->RunInference(batch.data(), count * kImSize_ * sizeof(float));
  }
  kInfer_->sync();
}



float ExperimentServer::TimeNoLoad(
    const std::vector<CompressedImage>& kCompressedImages,
    std::vector<float> &output) {
  auto start = std::chrono::high_resolution_clock::now();
  RunInferenceOnCompressed(kCompressedImages, output);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  float time = diff.count() / 1000.0;
  return time;
}

float ExperimentServer::TimeInferenceOnly() {
  const size_t kNbBatches = 1000;
  std::vector<float> batch;
  batch.reserve(kBatchSize_ * kImSize_);

  auto start = std::chrono::high_resolution_clock::now();

  #pragma omp parallel for
  for (size_t i = 0; i < kNbBatches; i++) {
    kInfer_->RunInference(batch.data(), kBatchSize_ * kImSize_ * sizeof(float));
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return diff.count() / 1000.0;
}

float ExperimentServer::TimeDecodePreprocOnly(const std::vector<CompressedImage>& kCompressedImages) {
  auto start = std::chrono::high_resolution_clock::now();

  std::vector<float> batch;
  batch.reserve(kBatchSize_ * kImSize_);

  #pragma omp parallel for
  for (size_t i = 0; i < kCompressedImages.size(); i += kBatchSize_) {
    for (size_t j = 0; j < kBatchSize_; j++) {
      if (i + j < kCompressedImages.size()) {
        kLoader_.DecodeAndPreproc(
            kCompressedImages[i + j],
            batch.data() + j * kImSize_);
      }
    }
  }


  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> diff = end - start;
  return diff.count() / 1000.0;
}
