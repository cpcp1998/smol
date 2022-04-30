#include <iostream>
#include <string>

#include "include/ipc.h"


int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cerr << "Usage: check_queue <queue name>" << std::endl;
    return 1;
  }
  const char *queueName = argv[1];

  MessageQueueShm mq(queueName, 0, false);
  mq.check();

  return 0;
}
