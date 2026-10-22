import urllib.request
import json

def test_api():
    sample_path = r"C:\Users\Hi\.gemini\antigravity\scratch\tesseract_htr_finetune\sample_test_data\synth_line_00000.png"
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="image"; filename="sample.png"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode("utf-8") + img_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request("http://127.0.0.1:5000/predict", data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode())
        print("API Response:", result)

if __name__ == "__main__":
    test_api()
