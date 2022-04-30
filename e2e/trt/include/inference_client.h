#ifndef INFERENCE_CLIENT_H_
#define INFERENCE_CLIENT_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>

#include "common.h"
#include "ipc.h"


class InferenceClient {
 private:
  MessageQueueShm inputQueue;
  MessageQueueShm outputQueue;
  std::atomic_size_t inputCount, outputCount;
  std::atomic_bool finished;
  std::thread receiver;
 public:
  InferenceClient(
      const std::string &inputQueueName,
      const std::string &outputQueueName) :
      inputQueue(inputQueueName.c_str(), 0, false),
      outputQueue(outputQueueName.c_str(), 0, false),
      inputCount(0), outputCount(0), finished(false),
      receiver([this] { discardResult(); }) {}

  void RunInference(void *data, size_t size) {
    void *msg = inputQueue.alloc(size);
    if (msg) {
      memcpy(msg, data, size);
      inputQueue.finishWrite(msg);
      ++inputCount;
    } else {
      std::cerr << "Message too large" << std::endl;
    }
  }

  void sync() {
    finished = true;
    receiver.join();
    finished = false;
    receiver = std::thread([this] { discardResult(); });
  }

  void discardResult() {
    while (!finished || inputCount < outputCount) {
      size_t size;
      void *addr = outputQueue.startRead(&size);
      outputQueue.finishRead(addr);
      ++outputCount;
    }
}
};

#endif // INFERENCE_CLIENT_H_
