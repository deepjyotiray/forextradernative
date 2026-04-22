"""Expose localhost:8899 to the internet via ngrok."""
from pyngrok import ngrok

tunnel = ngrok.connect(8899)
print(f"\n  Public URL: {tunnel.public_url}\n")
print("  Press Ctrl+C to stop the tunnel.\n")
ngrok.get_tunnels()

try:
    input()
except KeyboardInterrupt:
    ngrok.kill()
