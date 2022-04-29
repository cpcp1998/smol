#include <iostream>
#include <string>

#include "include/ipc.h"


int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: create_queue <queue name> <queue size>" << std::endl;
    return 1;
  }
  const char *queueName = argv[1];
  size_t queueSize = atoi(argv[2]);

  new MessageQueueShm(queueName, queueSize, true);

  return 0;
}
