#include <iostream>
#include <string>

#include "include/inference_server.h"


int main(int argc, char *argv[]) {
  assert(argc == 4);
  const char *inputQueueName = argv[1];
  std::string kEnginePath = argv[2];
  size_t kBatchSize = atoi(argv[3]);
  bool kDoMemcpy = true;

  OnnxInferenceServer inferServer{inputQueueName, kEnginePath, kBatchSize, kDoMemcpy};

  return 0;
}
