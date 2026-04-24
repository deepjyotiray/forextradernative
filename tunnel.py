"""Start named cloudflared tunnel for permanent DNS hosting."""
import subprocess, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CF_EXE = os.path.join(BASE_DIR, "cloudflared.exe")
CF_CONFIG = os.path.join(BASE_DIR, "cloudflared-config.yml")


def main():
    if not os.path.exists(CF_CONFIG):
        print(f"  ERROR: {CF_CONFIG} not found.")
        print("  Run: cloudflared tunnel login")
        print("       cloudflared tunnel create forextrader")
        print("  Then edit cloudflared-config.yml with your tunnel UUID and domain.")
        sys.exit(1)

    print("  Starting named Cloudflare tunnel...")
    print("  Your site will be at the hostname in cloudflared-config.yml")
    print("  Press Ctrl+C to stop.\n")

    proc = subprocess.Popen(
        [CF_EXE, "tunnel", "--config", CF_CONFIG, "run"],
        stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True,
    )

    try:
        for line in proc.stderr:
            sys.stderr.write(line)
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        print("\n  Tunnel stopped.")


if __name__ == "__main__":
    main()
