#include "common.h"

#define MUTEX_NAME "/sdasan/ipc-mutex"
#define MEM_NAME "/sdasan/ipc-mem"

class CPayload {
public:
  CPayload(int size) {
    init_payload(1024);
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
    m_locked = locked;
    m_name = name;
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

class CServerMutex : CMutex {
public:
  CServerMutex(const char* name): CMutex(name, true, true) {}
};

class CClientMutex : CMutex {
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

  ~CLock() {
    if (m_locked)
      m_mutex.unlock();
  }
};

class CSharedMemory {
private:
  void* m_map;
  bool m_create;

public:
  CSharedMemory(const char* name, bool create) {
    m_create = create;
    if (create)
      m_map = get_map(name, false);
    else
      m_map = get_map(name, true);
  }
};

void ipc() {
  CPayload payload(1024);
  //CServerMutex serverMutex(MUTEX_NAME);
  //CServerSharedMemory serverMemory(MEM_NAME);
}
