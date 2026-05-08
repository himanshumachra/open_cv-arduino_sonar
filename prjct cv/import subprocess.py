import subprocess
import datetime

# --- Configuration ---
PORT = "COM3"            # Change to your Arduino port
BOARD = "m328p"         # m328p = Uno/Nano, m2560 = Mega
PROGRAMMER = "arduino"  # Standard for most Arduinos
BAUD = "115200"         # Upload speed
OUTPUT_FILE = f"arduino_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.hex"

def backup_arduino():
    print(f"Attempting to capture firmware from {PORT}...")
    
    # Construct the avrdude command
    command = [
        "avrdude",
        "-p", BOARD,
        "-c", PROGRAMMER,
        "-P", PORT,
        "-b", BAUD,
        "-U", f"flash:r:{OUTPUT_FILE}:i"
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Successfully captured! File saved as: {OUTPUT_FILE}")
        else:
            print("Error capturing firmware:")
            print(result.stderr)
    except FileNotFoundError:
        print("Error: avrdude not found. Make sure it's in your System PATH.")

if __name__ == "__main__":
    backup_arduino()