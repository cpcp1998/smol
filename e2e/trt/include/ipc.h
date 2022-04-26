#pragma once

#include <stddef.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <pthread.h>
#include <stdio.h>
#include <iostream>

class SharedMemory {
public:
    SharedMemory(const char *name, size_t size, bool owner);

    ~SharedMemory();

    void *addr() const { return addr_; }
    size_t size() const { return size_; }
    bool owner() const { return owner_; }

private:
    char *name_;
    void *addr_;
    size_t size_;
    bool owner_;
};

/* ring buffer */
struct MessageQueueMessage {
    bool alloc;  // whether the message exists
    bool valid;  // whether the data field carries meaningful data
    size_t size;  // the size of the message, excluding the header
    char data[1];

    size_t totalSize() const {
        return size + headerSize();
    }

    static size_t headerSize() {
        return offsetof(MessageQueueMessage, data);
    }

    static MessageQueueMessage *getHeader(void *data) {
        return (MessageQueueMessage *)((char *)data - headerSize());
    }
};

struct MessageQueue {
    pthread_mutex_t mutex;
    pthread_cond_t rcond, wcond, fcond;
    size_t size;
    size_t readOff, transferOff, writeOff, freeOff;
    bool closed;
    char data[1];

    void init(size_t size);

    size_t totalSize() const {
        return size + offsetof(MessageQueue, data);
    }

    MessageQueueMessage *getMessage(size_t offset) {
        return (MessageQueueMessage *) &data[offset];
    }

    size_t getOffset(MessageQueueMessage *message) {
        return (char *)message - data;
    }

    size_t continuousFree() const {
        if (freeOff >= readOff) {
            return size - freeOff > readOff ? size - freeOff : readOff - 1;
        } else {
            return readOff - freeOff - 1;
        }
    }

    size_t allocate(size_t msgSize);

    void finishWrite(size_t offset);

    size_t startRead();

    void finishRead(size_t offset);

    void flush();

    void close();

    void destroy();
};

class MessageQueueShm {
public:
    MessageQueueShm(const char *name, size_t size, bool owner)
        : m_(name, size, owner), owner_(owner) {
        mq_ = (MessageQueue *) m_.addr();
        if (owner) {
            mq_->init(size);
        }
    }

    virtual ~MessageQueueShm() {
        if (owner_) mq_->destroy();
    }

    void *alloc(size_t size) {
        size_t offset = mq_->allocate(size);
        if (offset == -1) return nullptr;
        return mq_->getMessage(offset)->data;
    }

    void finishWrite(void *message) {
        size_t offset = mq_->getOffset(MessageQueueMessage::getHeader(message));
        mq_->finishWrite(offset);
    }

    void *startRead(size_t *size) {
        size_t offset = mq_->startRead();
        if (offset == -1) return nullptr;
        if (size) *size = mq_->getMessage(offset)->size;
        return mq_->getMessage(offset)->data;
    }

    void finishRead(void *message) {
        size_t offset = mq_->getOffset(MessageQueueMessage::getHeader(message));
        mq_->finishRead(offset);
    }

    void flush() {
        mq_->flush();
    }

    void close() {
        mq_->close();
    }

private:
    SharedMemory m_;
    MessageQueue *mq_;
    bool owner_;
};
