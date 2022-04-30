#include <iostream>
#include <string>

#include "include/inference_server.h"


int main(int argc, char *argv[]) {
  assert(argc == 5);
  std::string inputQueueName = argv[1];
  std::string outputQueueName = argv[2];
  std::string kEnginePath = argv[3];
  size_t kBatchSize = atoi(argv[4]);
  bool kDoMemcpy = true;

  OnnxInferenceServer inferServer{inputQueueName, outputQueueName, kEnginePath, kBatchSize, kDoMemcpy};

  return 0;
}
