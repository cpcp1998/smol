#ifndef INFERENCE_CLIENT_H_
#define INFERENCE_CLIENT_H_

#include <memory>
#include <stdint.h>
#include <string>
#include <thread>
#include <vector>
#include <iostream>

#include "common.h"
#include "ipc.h"


class InferenceClient {
 private:
  MessageQueueShm mq;
 public:
  InferenceClient(const char *inputQueueName) : mq(inputQueueName, 0, false) {}

  void RunInference(void *data, size_t size) {
    void *msg = mq.alloc(size);
    if (msg) {
      memcpy(msg, data, size);
      mq.finishWrite(msg);
    }
  }
};

#endif // INFERENCE_CLIENT_H_
