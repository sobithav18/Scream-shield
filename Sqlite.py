import sqlite3
import cv2
import pywhatkit
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


# Database setup
def setup_database():
    conn = sqlite3.connect('emergency_contacts.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            phone TEXT NOT NULL,
            emergency BOOLEAN DEFAULT 0
        )
    ''')
    cursor.execute('INSERT OR IGNORE INTO contacts (id, name, phone, emergency) VALUES (1, "nana", "+919652126186", 1)')
    cursor.execute('INSERT OR IGNORE INTO contacts (id, name, phone, emergency) VALUES (2, "amma", "+919247290263", 1)')
    conn.commit()
    conn.close()


# Fetch all emergency contacts
def get_emergency_contacts():
    conn = sqlite3.connect('emergency_contacts.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, phone FROM contacts WHERE emergency = 1')
    contacts = cursor.fetchall()
    conn.close()
    return contacts


# Capture one picture
def capture_picture():
    camera = cv2.VideoCapture(0)  # Open the default camera
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        return None

    ret, frame = camera.read()
    if ret:
        image_path = "scream_detected.jpg"
        cv2.imwrite(image_path, frame)  # Save the captured frame
        print(f"Picture saved: {image_path}")
        camera.release()
        return image_path
    else:
        print("Error: Could not capture a picture.")
        camera.release()
        return None


# Set up Selenium WebDriver with saved session cookies
def setup_driver_with_cookies():
    chrome_options = Options()
    chrome_options.add_argument(
        "--user-data-dir=chrome_data")  # Use a custom user data directory to reuse session cookies
    chrome_options.add_argument("--profile-directory=Default")  # Use the default profile

    # Path to the chromedriver (make sure it's compatible with your Chrome version)
    driver_path = '/Users/sobithav/Downloads/chromedriver-mac-arm64'  # Update this path to your chromedriver's location
    try:
        service = Service(driver_path)
        driver = webdriver.Chrome(service=service, options=chrome_options)
        driver.get('https://web.whatsapp.com')

        # Wait until the page is loaded and the "Type a message" field is visible
        WebDriverWait(driver, 60).until(
            EC.presence_of_element_located((By.XPATH, '//div[@title="Type a message"]'))
        )
        print("WhatsApp Web loaded successfully.")
        return driver
    except Exception as e:
        print(f"Error setting up the WebDriver: {e}")
        return None


# Send WhatsApp alert
def send_whatsapp_alert(contacts, image_path):
    driver = setup_driver_with_cookies()  # Set up driver with saved session cookies
    if driver:
        for name, phone in contacts:
            try:
                message = f"🚨 Emergency Alert! 🚨\n\nA scream was detected. Please check immediately. (For {name})"
                # Send image with message to the contact
                pywhatkit.sendwhats_image(phone, image_path, caption=message, wait_time=10, tab_close=True)
                print(f"Alert sent to {name} ({phone}) with picture.")
            except Exception as e:
                print(f"Error sending alert to {name} ({phone}): {e}")
        driver.quit()  # Close the driver after sending the alerts
    else:
        print("Failed to set up the driver. Cannot send messages.")


# Main function
def main():
    setup_database()  # Ensure the database and table are set up
    contacts = get_emergency_contacts()  # Fetch emergency contacts

    print("Simulating scream detection... Press Ctrl+C to stop.")
    try:
        # Simulate scream detection
        print("Simulated scream detected!")
        image_path = capture_picture()  # Capture one picture
        if image_path:  # If a picture was successfully captured
            send_whatsapp_alert(contacts, image_path)  # Send alerts
        else:
            print("No picture captured. Alert not sent.")
    except KeyboardInterrupt:
        print("Stopped by user.")


if __name__ == "__main__":
    main()
