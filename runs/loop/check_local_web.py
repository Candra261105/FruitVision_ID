from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:5173", wait_until="networkidle")
    print("title=" + page.title())
    print("body_length=" + str(len(page.locator("body").inner_text())))
    print("url=" + page.url)
    browser.close()
