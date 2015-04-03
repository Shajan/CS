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

  bool validate();
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
