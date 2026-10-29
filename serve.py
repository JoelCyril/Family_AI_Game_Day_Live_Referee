"""Family AI Game Day — Live Referee Server Launcher.

Hosts the referee application on 0.0.0.0:8000 with optional HTTPS support
so mobile devices (iOS Safari, Android Chrome) can access the live camera
over LAN Wi-Fi without security restrictions.
"""
import argparse
import datetime
import ipaddress
import os
import socket
import sys
from pathlib import Path
import uvicorn

ROOT = Path(__file__).parent

def get_lan_ip() -> str:
    """Detect the host machine's primary local network IPv4 address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()

def ensure_ssl_certs(lan_ip: str, cert_file: Path, key_file: Path):
    """Generate self-signed SSL certificate with SAN if not already present."""
    if cert_file.exists() and key_file.exists():
        return

    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization

    print(f"[*] Generating SSL certificate for {lan_ip}...")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, 'Family AI Game Day Referee')
    ])

    san_list = [
        x509.DNSName('localhost'),
        x509.IPAddress(ipaddress.ip_address('127.0.0.1'))
    ]
    try:
        san_list.append(x509.IPAddress(ipaddress.ip_address(lan_ip)))
    except ValueError:
        pass

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_list), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_file, 'wb') as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open(cert_file, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print("[*] SSL certificate created successfully.")

def print_qr(url: str):
    """Print ASCII QR code to terminal for easy phone scanning."""
    try:
        import qrcode
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        print("\nScan this QR code with your phone camera:")
        qr.print_ascii(invert=True)
    except Exception:
        pass

def main():
    parser = argparse.ArgumentParser(description="Host Family AI Game Day Live Referee")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--no-ssl", action="store_true", help="Run without SSL (plain HTTP)")
    args = parser.parse_args()

    lan_ip = get_lan_ip()
    cert_path = ROOT / "cert.pem"
    key_path = ROOT / "key.pem"

    use_ssl = not args.no_ssl
    if use_ssl:
        ensure_ssl_certs(lan_ip, cert_path, key_path)
        protocol = "https"
    else:
        protocol = "http"

    local_url = f"{protocol}://127.0.0.1:{args.port}"
    phone_url = f"{protocol}://{lan_ip}:{args.port}"

    print("=" * 64, flush=True)
    print("  FAMILY AI GAME DAY — LIVE AI REFEREE (ADVANCED TRACK)", flush=True)
    print("  Abu Dhabi University Challenge", flush=True)
    print("=" * 64, flush=True)
    print(f"  Laptop / Local URL:  {local_url}", flush=True)
    print(f"  Phone / Mobile URL:  {phone_url}", flush=True)
    print("=" * 64, flush=True)

    if use_ssl:
        print("\n[!] MOBILE PHONE INSTRUCTIONS (iOS Safari & Android Chrome):", flush=True)
        print(f"  1. Ensure your phone and PC are on the SAME Wi-Fi network.", flush=True)
        print(f"  2. Open {phone_url} on your phone.", flush=True)
        print(f"  3. Tap 'Advanced' / 'Details' -> 'Proceed to {lan_ip}' (self-signed cert).", flush=True)
        print(f"  4. Tap 'Start Camera' and allow camera access. Use '🔄 Flip Camera' for rear lens!", flush=True)

    print_qr(phone_url)
    print(f"\nStarting server on 0.0.0.0:{args.port}...", flush=True)

    ssl_kwargs = {}
    if use_ssl:
        ssl_kwargs = {
            "ssl_keyfile": str(key_path),
            "ssl_certfile": str(cert_path)
        }

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=args.port,
        app_dir=str(ROOT),
        **ssl_kwargs
    )

if __name__ == "__main__":
    main()
