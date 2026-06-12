import urllib.request, json, uuid, os, time

filepath = r"C:\Users\HP\Desktop\Deepfake Defence\ai-engine\data\140k_extracted\real_vs_fake\real-vs-fake\test\fake\00276TOPP4.jpg"
boundary = uuid.uuid4().hex

with open(filepath, "rb") as f:
    file_data = f.read()

body = (
    "--" + boundary + "\r\n"
    + 'Content-Disposition: form-data; name="file"; filename="00276TOPP4.jpg"\r\n'
    + "Content-Type: image/jpeg\r\n\r\n"
).encode("utf-8") + file_data + ("\r\n--" + boundary + "--\r\n").encode("utf-8")

req = urllib.request.Request("http://127.0.0.1:8000/api/v1/explain?include_lime=false", data=body, method="POST")
req.add_header("Content-Type", "multipart/form-data; boundary=" + boundary)

t0 = time.time()
try:
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read().decode("utf-8"))
    print(f"Explain Time: {time.time()-t0:.2f}s")
    print(json.dumps(data, indent=2))
except Exception as e:
    print(f"Failed: {e}")
