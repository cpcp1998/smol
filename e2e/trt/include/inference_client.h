#ifndef INFERENCE_CLIENT_H_
#define INFERENCE_CLIENT_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>
#include <iostream>
#include <mutex>
#include <condition_variable>

#include "common.h"
#include "ipc.h"


class InferenceClient {
 private:
  MessageQueueShm inputQueue;
  MessageQueueShm outputQueue;
  size_t inputCount, outputCount;
  bool finished;
  std::thread receiver;
  std::mutex mutex;
  std::condition_variable cond;
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
      mutex.lock();
      ++inputCount;
      cond.notify_one();
      mutex.unlock();
    } else {
      std::cerr << "Message too large" << std::endl;
    }
  }

  void sync() {
    mutex.lock();
    finished = true;
    cond.notify_one();
    mutex.unlock();
    receiver.join();
    finished = false;
    receiver = std::thread([this] { discardResult(); });
  }

  void discardResult() {
    while (true) {
      std::unique_lock<std::mutex> lock(mutex);
      while (!finished && inputCount <= outputCount) cond.wait(lock);
      if (finished && inputCount == outputCount) break;
      size_t size;
      void *addr = outputQueue.startRead(&size);
      outputQueue.finishRead(addr);
      ++outputCount;
      lock.unlock();
    }
}
};

#endif // INFERENCE_CLIENT_H_
