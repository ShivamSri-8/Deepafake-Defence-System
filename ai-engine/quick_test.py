import urllib.request, json, uuid, sys

filepath = r"C:\Users\HP\Desktop\Deepfake Defence\ai-engine\data\140k_extracted\real_vs_fake\real-vs-fake\test\real\00001.jpg"
label = "REAL"

if len(sys.argv) > 1 and sys.argv[1] == "fake":
    filepath = r"C:\Users\HP\Desktop\Deepfake Defence\ai-engine\data\140k_extracted\real_vs_fake\real-vs-fake\test\fake\00276TOPP4.jpg"
    label = "FAKE"

boundary = uuid.uuid4().hex
with open(filepath, "rb") as f:
    file_data = f.read()

import os
filename = os.path.basename(filepath)

body = (
    "--" + boundary + "\r\n"
    + 'Content-Disposition: form-data; name="file"; filename="' + filename + '"\r\n'
    + "Content-Type: image/jpeg\r\n"
    + "\r\n"
).encode("utf-8") + file_data + ("\r\n--" + boundary + "--\r\n").encode("utf-8")

req = urllib.request.Request("http://127.0.0.1:8000/api/v1/detect", data=body, method="POST")
req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)
resp = urllib.request.urlopen(req, timeout=60)
data = json.loads(resp.read().decode("utf-8"))

print(f"\n=== {label} IMAGE: {filename} ===")
print(json.dumps(data, indent=2))
