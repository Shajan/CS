#include <stdlib.h> // exit
#include <unistd.h> // fork
#include "common.h"

#define MUTEX_NAME "/sdasan/ipc-mutex"
#define MEM_NAME "/sdasan/ipc-mem"
#define SIZE 1024

class CPayload {
public:
  CPayload(int size) {
    init_payload(size);
  }
  ~CPayload() {
    free_payload();
  }
};

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
  bool m_locked;
  CMutex& m_mutex;
public:
  CLock(CMutex& m) : m_mutex(m) {
    m_mutex.lock();
    m_locked = true;
  }

  CLock(CMutex& m, int timeoutSeconds) : m_mutex(m) {
    m_locked = m_mutex.lock(timeoutSeconds);
  }

  void release() {
    if (m_locked)
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

void ipc() {
  CPayload payload(SIZE);
  CServerMutex serverMutex(MUTEX_NAME);
  CSharedMemory serverMemory(MEM_NAME, SIZE, true);

  CLock lock(serverMutex);
  pid_t childpid;
  if ((childpid = fork()) == -1) {
    sys_error_exit("fork");
  }

  if (childpid == 0) {
    /* Child process */
    CPayload payload(SIZE);
    CClientMutex clientMutex(MUTEX_NAME);
    CSharedMemory clientMemory(MEM_NAME, SIZE, false);
    exit(0);
  } else {
    /* Parent process */
    sleep(2);
  }
}
