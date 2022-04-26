#include "ipc.h"

SharedMemory::SharedMemory(const char *name, size_t size, bool owner)
    : name_(strdup(name)), addr_(NULL), size_(size), owner_(owner) {
    int fd;
    if (owner_) {
        mode_t umask_orig = umask(0);
        fd = shm_open(name_, O_RDWR | O_CREAT | O_EXCL, 0666);
        umask(umask_orig);
        if (fd < 0) {
            perror("SharedMemory");
            return;
        }
        if (ftruncate(fd, size_) < 0) {
            perror("SharedMemory");
            return;
        }
    } else {
        fd = shm_open(name_, O_RDWR, 0);
        if (fd < 0) {
            perror("SharedMemory");
            return;
        }
    }
    struct stat buf;
    if (fstat(fd, &buf) < 0) {
        perror("SharedMemory");
        return;
    }
    size_ = buf.st_size;
    addr_ = mmap(NULL, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr_ == MAP_FAILED) {
        addr_ = NULL;
        perror("SharedMemory");
    }
}

SharedMemory::~SharedMemory() {
    if (addr_) munmap(addr_, size_);
    if (addr_ && owner_) shm_unlink(name_);
    free(name_);
}


void MessageQueue::init(size_t size) {
    pthread_mutexattr_t mutexattr;
    pthread_condattr_t condattr;
    pthread_mutexattr_init(&mutexattr);
    pthread_mutexattr_setpshared(&mutexattr, PTHREAD_PROCESS_SHARED);
    pthread_condattr_init(&condattr);
    pthread_condattr_setpshared(&condattr, PTHREAD_PROCESS_SHARED);
    pthread_mutex_init(&mutex, &mutexattr);
    pthread_cond_init(&rcond, &condattr);
    pthread_cond_init(&wcond, &condattr);
    pthread_cond_init(&fcond, &condattr);
    pthread_mutexattr_destroy(&mutexattr);
    pthread_condattr_destroy(&condattr);
    this->size = size - offsetof(MessageQueue, data);
    readOff = transferOff = writeOff = freeOff = 0;
    closed = false;
}

size_t MessageQueue::allocate(size_t msgSize) {
    size_t offset;
    size_t allocSize = MessageQueueMessage::headerSize() + msgSize;
    if (closed) return -1;
    if (allocSize > size) return -1;
    pthread_mutex_lock(&mutex);
    while (continuousFree() < allocSize) {
        pthread_cond_wait(&wcond, &mutex);
    }
    if (freeOff >= readOff && size - freeOff < allocSize) {
        getMessage(freeOff)->alloc = false;
        offset = 0;
        if (writeOff == freeOff) writeOff = 0;
        if (transferOff == freeOff) transferOff = 0;
        if (readOff == freeOff) readOff = 0;
    } else {
        offset = freeOff;
    }
    freeOff = offset + allocSize;
    if (freeOff == size) freeOff = 0;
    MessageQueueMessage *message = getMessage(offset);
    message->size = msgSize;
    message->alloc = true;
    message->valid = false;
    pthread_cond_signal(&wcond);
    pthread_cond_signal(&fcond);
    pthread_mutex_unlock(&mutex);
    return offset;
}

void MessageQueue::finishWrite(size_t offset) {
    pthread_mutex_lock(&mutex);
    getMessage(offset)->valid = true;
    while (writeOff != freeOff) {
        if (!getMessage(writeOff)->valid) break;
        writeOff += getMessage(writeOff)->totalSize();
        if (writeOff == size || writeOff != freeOff && !getMessage(writeOff)->alloc)
            writeOff = 0;
    }
    pthread_cond_signal(&rcond);
    pthread_cond_signal(&fcond);
    pthread_mutex_unlock(&mutex);
}

size_t MessageQueue::startRead() {
    pthread_mutex_lock(&mutex);
    while (transferOff == writeOff && !closed) {
        pthread_cond_wait(&rcond, &mutex);
    }
    size_t offset;
    if (transferOff == writeOff) {
        offset = -1;
    } else {
        offset = transferOff;
        getMessage(offset)->valid = false;
        transferOff += getMessage(offset)->totalSize();
        if (transferOff == size || transferOff != writeOff && !getMessage(transferOff)->alloc) {
            transferOff = 0;
        }
    } 
    pthread_cond_signal(&rcond);
    pthread_cond_signal(&fcond);
    pthread_mutex_unlock(&mutex);
    return offset;
}

void MessageQueue::finishRead(size_t offset) {
    pthread_mutex_lock(&mutex);
    getMessage(offset)->valid = true;
    while (readOff != transferOff) {
        if (!getMessage(readOff)->valid) break;
        readOff += getMessage(readOff)->totalSize();
        if (readOff == size || readOff != transferOff && !getMessage(readOff)->alloc)
            readOff = 0;
    }
    if (readOff == freeOff) {
        readOff = transferOff = writeOff = freeOff = 0;
    }
    pthread_cond_signal(&wcond);
    pthread_cond_signal(&fcond);
    pthread_mutex_unlock(&mutex);
}

void MessageQueue::flush() {
    pthread_mutex_lock(&mutex);
    while (readOff != freeOff) {
        pthread_cond_wait(&fcond, &mutex);
    }
    pthread_mutex_unlock(&mutex);
}

void MessageQueue::close() {
    pthread_mutex_lock(&mutex);
    closed = true;
    pthread_cond_wait(&rcond, &mutex);
    pthread_mutex_unlock(&mutex);
}

void MessageQueue::destroy() {
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&rcond);
    pthread_cond_destroy(&wcond);
    pthread_cond_destroy(&fcond);
}
