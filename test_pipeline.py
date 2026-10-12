import os
import requests
from edge_processor import EdgeProcessor

CLOUD_URL = "http://127.0.0.1:5000"
SAMPLE_DIR = "sample_images"

def run_pipeline():
    # 1. Clear previous logs and image storage on server reset
    try:
        requests.post(f"{CLOUD_URL}/clear_logs")
        print("Connected to Cloud Server: Cleared existing logs & storage.")
    except Exception as e:
        print(f"Could not connect to cloud server: {e}")
        return

    # 2. Instantiate edge processor with aligned two-tier parameters
    processor = EdgeProcessor(
    blur_threshold=18.0,
    phash_threshold=2,
    entropy_threshold=3.5
    )
    processor.reset()

    if not os.path.exists(SAMPLE_DIR):
        print(f"Directory '{SAMPLE_DIR}' does not exist.")
        return

    image_files = sorted(os.listdir(SAMPLE_DIR))

    print("\n==============================================")
    print(" STARTING EDGE WILDLIFE MONITORING SIMULATION ")
    print("==============================================\n")

    for filename in image_files:
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.avif')):
            continue

        filepath = os.path.join(SAMPLE_DIR, filename)

        # Execute edge filtering pipeline
        status, blur_score, reason = processor.process_frame(filepath)

        print(f"--- Processing: {filename} ---")
        print(f"  ├─ Blur Score: {blur_score}")
        print(f"  ├─ Status: {status}")
        print(f"  └─ Action: {reason}")

        # Send decision logs to Flask server
        log_data = {
            "filename": filename,
            "blur_score": str(blur_score),
            "status": status,
            "reason": reason
        }

        try:
            requests.post(f"{CLOUD_URL}/log", json=log_data)
        except Exception as e:
            print(f"  [Server Error] Could not send log: {e}")

        # Transmit image file to cloud gallery only if PASSED
        if status == "PASSED":
            try:
                with open(filepath, 'rb') as f:
                    files = {'file': (filename, f, 'image/jpeg')}
                    requests.post(f"{CLOUD_URL}/upload", files=files)
            except Exception as e:
                print(f"  [Server Error] Could not upload image: {e}")

if __name__ == "__main__":
    run_pipeline()