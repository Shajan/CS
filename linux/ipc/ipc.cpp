#include <stdlib.h> // exit
#include <string.h> // memcpy
#include <unistd.h> // fork
#include "common.h"
#include "message.h"

#define MUTEX_NAME "/sdasan/ipc-mutex"
#define MEM_NAME "/sdasan/ipc-mem"
#define SIZE 1024

class CMutex {
private:
  mutex m_mutex;
  const char* m_name;
  bool m_create;
  bool m_locked; //just a hint

public:
  CMutex(const char* name, bool locked, bool create) {
    m_name = name;
    m_locked = locked;
    m_create = create;
    if (create)
      m_mutex = create_mutex(m_name, m_locked);
    else
      m_mutex = open_mutex(name);
  }

  void lock() {
    lock_mutex(m_mutex);
    m_locked = true;
  }

  bool try_lock() {
    return m_locked = trylock_mutex(m_mutex);
  }

  bool lock(int timeout_seconds) {
    return m_locked = lock_withtimeout_mutex(m_mutex, timeout_seconds);
  }

  void unlock() {
    if (!m_locked)
      sys_warn("CMutex:unlock potentialy unlocking mutex while it's not locked %s", m_name);
    unlock_mutex(m_mutex);
    m_locked = false;
  }

  virtual ~CMutex() {
    if (m_locked)
      sys_warn("~CMutex: potentialy %s mutex while it's locked %s",
        (m_create ? "destroying" : "closing"), m_name);
    if (m_create)
      destroy_mutex(m_name, m_mutex);
    else
      close_mutex(m_mutex);
  }
};

class CServerMutex : public CMutex {
public:
  CServerMutex(const char* name): CMutex(name, false, true) {}
};

class CClientMutex : public CMutex {
public:
  CClientMutex(const char* name): CMutex(name, false, false) {}
};

class CLock {
private:
  CMutex& m_mutex;
  bool m_locked;
public:
  CLock(CMutex& m) : m_mutex(m) {
    m_mutex.lock();
    m_locked = true;
  }

  CLock(CMutex& m, int timeoutSeconds) : m_mutex(m) {
    m_locked = m_mutex.lock(timeoutSeconds);
  }

  void lock() {
    m_mutex.lock();
    m_locked = true;
  }

  bool lock(int timeoutSeconds) {
    return m_locked = m_mutex.lock(timeoutSeconds);
  }

  void unlock() {
    m_mutex.unlock();
    m_locked = false;
  }

  ~CLock() {
    if (m_locked)
      m_mutex.unlock();
  }
};

class CSharedMemory {
private:
  map m_map;
  int m_size;
public:
  CSharedMemory(const char* name, int size, bool create) {
    m_size = size;
    m_map = get_map(name, m_size, create);
  }
  ~CSharedMemory() {
    unmap(m_map, m_size);
  }
};

#define ADD 1
#define MUL 2

class ISerialize {
  void write(void* p);
  void read(void* p);
};

class CRequest : public ISerialize {
private:
  int m_a, m_b;
public:
  CRequest(void* p) { read(p); }
  CRequest(int a, int b) : m_a(a), m_b(b) {}

  void write(void* p) {
    memcpy(p, (void*)&m_a, sizeof(m_a));
    memcpy((char*)p + sizeof(m_a), (void*)&m_b, sizeof(m_b));
  }

  void read(void* p) {
    memcpy((void*)&m_a, p, sizeof(m_a));
    memcpy((void*)&m_b, (char*)p + sizeof(m_a), sizeof(m_b));
  }

  void log() { ::log("a:%d, b:%d", m_a, m_b); }
};

class CResponse : public ISerialize {
private:
  int m_result;
public:
  CResponse(int result) : m_result(result) {}
  CResponse(void* p) { read(p); }
  void write(void* p) { memcpy(p, (void*)&m_result, sizeof(m_result)); }
  void read(void* p) { memcpy((void*)&m_result, p, sizeof(m_result)); }
  void log() { ::log("result:%d", m_result); }
};

void server(CMutex& mutex, CSharedMemory& memory) {
}

void client(CMutex& mutex, CSharedMemory& memory) {
}

void ipc() {
  CServerMutex serverMutex(MUTEX_NAME);
  CSharedMemory serverMemory(MEM_NAME, CMessage::allocation_size(SIZE), true);

  CLock lock(serverMutex);
  pid_t childpid;
  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process */
    CClientMutex clientMutex(MUTEX_NAME);
    CSharedMemory clientMemory(MEM_NAME, CMessage::allocation_size(SIZE), false);
    server(clientMutex, clientMemory);
    exit(0);
  } else {
    /* Parent process */
    client(serverMutex, serverMemory);
    sleep(2);
  }
}
