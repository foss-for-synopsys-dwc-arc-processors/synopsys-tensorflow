#include "mwnn_inference_shm.h"

namespace metawarenn {

MWNNSharedMemory::MWNNSharedMemory() {
  std::cout << "\n MWNNSharedMemory Constructor!!";

  key_t key = ftok("/tmp/", 111);
  int shmid = shmget(key, TOTAL_MEMORY_SIZE, 0644|IPC_CREAT);
  if (shmid == -1) {
      perror("Shared memory");
      exit(1);
  }

  shmp = (char*)shmat(shmid, NULL, 0);
  if (shmp == (void *) -1) {
      perror("Shared memory attach");
      exit(1);
  }
}

} //metawarenn

