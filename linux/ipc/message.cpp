#include "common.h"
#include "message.h"

bool CMessage::validate() {
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
