#ifndef METAWARENN_SHM_H_
#define METAWARENN_SHM_H_

#include <iostream>
#include <sys/ipc.h>
#include <sys/shm.h>
#define TOTAL_MEMORY_SIZE 15000000

namespace metawarenn {

class MWNNSharedMemory {
  public:
    MWNNSharedMemory();
    char *shmp;
};

} //metawarenn

#endif //METAWARENN_SHM_H_
