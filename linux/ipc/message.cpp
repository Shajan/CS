#include <stdlib.h>
#include <string.h>
#include "common.h"

#define HEADER_CANARY 0xFACEF00D
#define FOOTER_CANARY 0xDEADBEEF

struct message_header {
  unsigned long canary;
  unsigned long session;
  unsigned long operation;
  unsigned long size;
  char data[0];
};

struct message_footer {
  unsigned long session;
  unsigned long operation;
  unsigned long size;
  unsigned long canary;
};

class CMessage {
private:
  message_header* m_header;
public:
  CMessage(void* memory) { m_header = (message_header*)memory; }

  CMessage(void* memory, unsigned long session, unsigned long operation, unsigned long size) {
    m_header = (message_header*)memory;
    m_header->canary = HEADER_CANARY;
    m_header->session = session;
    m_header->operation = operation;
    m_header->size = size;

    message_footer* footer = get_footer();
    footer->session = session;
    footer->operation = operation;
    footer->size = size;
    footer->canary = FOOTER_CANARY;
  }

  bool validate() {
    if (m_header->canary != HEADER_CANARY) {
      log_error("Header Canary : expected 0x%08X, found 0x%08X", HEADER_CANARY, m_header->canary);
      return false;
    }
    message_footer* footer = get_footer();
    if (footer->canary != FOOTER_CANARY) {
      log_error("Footer Canary : expected 0x%08X, found 0x%08X", FOOTER_CANARY, footer->canary);
      return false;
    }
    if (m_header->size != footer->size) {
      log_error("Size : header 0x%08X, footer 0x%08X", m_header->size, footer->size);
      return false;
    }
    if (m_header->session != footer->session) {
      log_error("Session : header 0x%08X, footer 0x%08X", m_header->session, footer->session);
      return false;
    }
    return true;
  }

  bool validate(unsigned long session) { return validate() && (m_header->session == session); }
  void* data() { return (void*) m_header->data; }
  unsigned long session() { return m_header->session; }
  unsigned long operation() { return m_header->operation; }

private:
  message_footer* get_footer() {
    return (message_footer*) ((char*)m_header + sizeof(message_header) + m_header->size);
  }

public:
  static unsigned long allocation_size(unsigned long data_size) {
    return sizeof(message_header) + data_size + sizeof(message_footer);
  }
};
