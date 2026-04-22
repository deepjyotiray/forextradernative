"""Start cloudflared tunnel, capture the public URL, save it, and push to GitHub."""
import subprocess, re, os, signal, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
URL_FILE = os.path.join(BASE_DIR, "cf_url.txt")
CF_EXE = os.path.join(BASE_DIR, "cloudflared.exe")


def git(cmd):
    subprocess.run(f"git {cmd}", cwd=BASE_DIR, shell=True, capture_output=True)


def push_url(url):
    with open(URL_FILE, "w") as f:
        f.write(url)
    git("add cf_url.txt")
    git('commit -m "update tunnel url"')
    git("push")
    print(f"  Pushed URL to GitHub: {url}")


def main():
    proc = subprocess.Popen(
        [CF_EXE, "tunnel", "--url", "http://127.0.0.1:8899"],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )

    url_pattern = re.compile(r"(https://[a-z0-9-]+\.trycloudflare\.com)")
    found = False

    try:
        for line in proc.stderr:
            sys.stderr.write(line)
            m = url_pattern.search(line)
            if m and not found:
                found = True
                push_url(m.group(1))
                print("\n  Tunnel running. Press Ctrl+C to stop.\n")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("\n  Tunnel stopped.")


if __name__ == "__main__":
    main()
