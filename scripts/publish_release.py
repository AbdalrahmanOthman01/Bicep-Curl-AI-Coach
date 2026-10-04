import os
import sys
import json
import subprocess
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

def get_github_token():
    # 1. Environment variable
    if os.environ.get("GITHUB_TOKEN"):
        return os.environ.get("GITHUB_TOKEN")
    if os.environ.get("GH_TOKEN"):
        return os.environ.get("GH_TOKEN")
    
    # 2. Git credential manager
    try:
        proc = subprocess.run(
            ["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n",
            capture_output=True,
            text=True,
            check=True
        )
        for line in proc.stdout.splitlines():
            if line.startswith("password="):
                return line.split("=", 1)[1].strip()
    except Exception as e:
        print(f"Git credential fill failed: {e}")
        
    # 3. Fail if not found
    return None

def make_github_request(url, method="GET", data=None, headers=None):
    req = urllib.request.Request(url, data=data, method=method)
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
            return resp.status, json.loads(content.decode("utf-8")) if content else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        print(f"HTTP Error {e.code}: {body}")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}

def publish_release(repo="AbdalrahmanOthman01/Bicep-Curl-AI-Coach", tag="v1.0.0", title="Bicep Curl AI Coach v1.0.0"):
    token = get_github_token()
    if not token:
        print("ERROR: No GitHub token found.")
        sys.exit(1)
        
    print(f"Authenticated with GitHub token ({token[:7]}...)")
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "BicepCurlReleasePublisher"
    }
    
    # Check if release already exists
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    status, data = make_github_request(url, method="GET", headers=headers)
    
    release_info = None
    if status == 200:
        print(f"Release {tag} already exists.")
        release_info = data
    else:
        print(f"Creating new release {tag}...")
        create_url = f"https://api.github.com/repos/{repo}/releases"
        body = {
            "tag_name": tag,
            "target_commitish": "main",
            "name": title,
            "body": """## Bicep Curl AI Coach - Release v1.0.0

### Highlights:
- **Biomechanical Upper-Body AI:** Real-time form detection trained with MediaPipe Pose and Random Forest classifier (94.86% test accuracy).
- **Machine Bicep Curl Optimized:** Specially formulated for bicep curl machines / preacher benches with zero leg dependency.
- **Real-Time Web Dashboard:** Interactive canvas with pose landmarks, SVG repetition progress ring, sound effects, and auditory feedback.
- **Standalone Windows Executable:** No Python or dependencies required on the host system. Simply download `BicepCurlAICoach.exe` and launch!

### Included Executable:
- `BicepCurlAICoach.exe`: Standalone portable launcher that boots the AI model server and opens the dashboard in your default browser.
""",
            "draft": False,
            "prerelease": False
        }
        status, release_info = make_github_request(
            create_url,
            method="POST",
            data=json.dumps(body).encode("utf-8"),
            headers={**headers, "Content-Type": "application/json"}
        )
        if status not in (200, 201):
            print(f"Failed to create release: {release_info}")
            sys.exit(1)
        print(f"Created release: {release_info.get('html_url')}")
        
    upload_url_template = release_info.get("upload_url", "")
    # upload_url is in format: https://uploads.github.com/repos/OWNER/REPO/releases/ID/assets{?name,label}
    clean_upload_url = upload_url_template.split("{")[0]
    
    exe_path = Path("dist/BicepCurlAICoach.exe")
    if not exe_path.exists():
        print(f"Error: {exe_path} does not exist yet. Please build it first.")
        sys.exit(1)
        
    asset_name = "BicepCurlAICoach.exe"
    asset_size = exe_path.stat().st_size
    print(f"Found binary: {exe_path} ({asset_size / (1024*1024):.2f} MB)")
    
    # Check if asset already exists in the release
    existing_assets = release_info.get("assets", [])
    for asset in existing_assets:
        if asset.get("name") == asset_name:
            print(f"Asset {asset_name} already exists (ID: {asset['id']}). Deleting old asset...")
            del_url = f"https://api.github.com/repos/{repo}/releases/assets/{asset['id']}"
            del_status, _ = make_github_request(del_url, method="DELETE", headers=headers)
            print(f"Deleted old asset status: {del_status}")
            
    print(f"Uploading {asset_name} to GitHub Release...")
    upload_target = f"{clean_upload_url}?name={urllib.parse.quote(asset_name)}"
    
    with open(exe_path, "rb") as f:
        file_bytes = f.read()
        
    upload_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "BicepCurlReleasePublisher",
        "Content-Type": "application/octet-stream",
        "Content-Length": str(len(file_bytes))
    }
    
    req = urllib.request.Request(upload_target, data=file_bytes, headers=upload_headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            print(f"Asset successfully uploaded!")
            print(f"Download URL: {resp_data.get('browser_download_url')}")
            print(f"Release URL: {release_info.get('html_url')}")
            return resp_data.get('browser_download_url'), release_info.get('html_url')
    except urllib.error.HTTPError as e:
        print(f"Failed to upload asset. HTTP {e.code}: {e.read().decode('utf-8')}")
        sys.exit(1)

if __name__ == "__main__":
    publish_release()
