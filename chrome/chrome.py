import pychrome
import os

# Get the debug port
debug_port = os.environ.get('CHROME_DEBUG_PORT', '30010')

# Create a browser instance
browser = pychrome.Browser(url=f"http://127.0.0.1:{debug_port}")

# Create a tab
tab = browser.new_tab()

def request_intercepted(**kwargs):
    print("Request Intercepted:", kwargs.get('request').get('url'))

try:
    # Start the tab
    tab.start()

    # Set up network event listeners
    tab.Network.requestIntercepted = request_intercepted
    tab.Network.enable()

    # Navigate to a page
    tab.Page.navigate(url="www.google.com", _timeout=5)

    # Wait for the page to load
    tab.wait(5)

    # Get cookies
    cookies = tab.Network.getCookies()
    print("Cookies:", cookies)

finally:
    # Stop the tab (closes it)
    tab.stop()

    # Close the browser
    browser.close()

