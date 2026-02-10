#!/usr/bin/env python3
"""
Script to obtain cTrader Open API Access Token.

This script helps you get an access token for cTrader Open API by:
1. Generating an authorization URL
2. Optionally starting a local server to receive the authorization code
3. Exchanging the authorization code for an access token

Usage:
    # Set environment variables
    export CTRADER_CLIENT_ID="your_client_id"
    export CTRADER_CLIENT_SECRET="your_client_secret"
    export CTRADER_REDIRECT_URI="http://localhost:8000/callback"

    # Run the script
    python get_ctrader_token.py

    # Or use the interactive mode
    python get_ctrader_token.py --interactive
"""
import os
import sys
import urllib.parse
import requests
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler to receive the authorization callback"""

    def __init__(self, *args, authorization_code_queue=None, **kwargs):
        self.authorization_code_queue = authorization_code_queue
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET request from cTrader redirect"""
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)

        if 'code' in query_params:
            code = query_params['code'][0]
            self.authorization_code_queue.append(code)

            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"""
            <html>
            <head><title>Authorization Successful</title></head>
            <body>
                <h1>Authorization Successful!</h1>
                <p>You can close this window and return to the terminal.</p>
                <p>The authorization code has been received.</p>
            </body>
            </html>
            """)
        else:
            error = query_params.get('error', ['Unknown error'])[0]
            self.send_response(400)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(f"""
            <html>
            <head><title>Authorization Failed</title></head>
            <body>
                <h1>Authorization Failed</h1>
                <p>Error: {error}</p>
            </body>
            </html>
            """.encode())

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass


def get_authorization_url(client_id: str, redirect_uri: str, scope: str = "trading") -> str:
    """Generate the authorization URL.

    Args:
        scope: Permission scope to request. Use 'trading' for full trade access
               or 'accounts' for view-only access.
    """
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'product': 'web'
    }
    query_string = urllib.parse.urlencode(params)
    return f"https://id.ctrader.com/my/settings/openapi/grantingaccess/?{query_string}"


def exchange_code_for_token(
    authorization_code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str
) -> dict:
    """Exchange authorization code for access token"""
    params = {
        'grant_type': 'authorization_code',
        'code': authorization_code,
        'redirect_uri': redirect_uri,
        'client_id': client_id,
        'client_secret': client_secret
    }

    url = f"https://openapi.ctrader.com/apps/token?{urllib.parse.urlencode(params)}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error exchanging code for token: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        raise


def start_callback_server(port: int, redirect_uri: str) -> list:
    """Start a local HTTP server to receive the authorization callback"""
    authorization_code_queue = []

    def handler_factory():
        def create_handler(*args, **kwargs):
            return CallbackHandler(*args, authorization_code_queue=authorization_code_queue, **kwargs)
        return create_handler

    server = HTTPServer(('localhost', port), handler_factory())

    def run_server():
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    return authorization_code_queue


def main():
    parser = argparse.ArgumentParser(
        description='Get cTrader Open API Access Token',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python get_ctrader_token.py --interactive

  # Manual mode - you'll need to paste the authorization code
  python get_ctrader_token.py --manual

  # With custom redirect URI
  python get_ctrader_token.py --redirect-uri http://localhost:9000/callback
        """
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start a local server to automatically receive the authorization code'
    )
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Manual mode - you will paste the authorization code yourself'
    )
    parser.add_argument(
        '--redirect-uri',
        type=str,
        help='Custom redirect URI (must match the one registered with cTrader)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port for the callback server (default: 8000)'
    )
    parser.add_argument(
        '--scope',
        type=str,
        default='trading',
        choices=['trading', 'accounts'],
        help='Permission scope: "trading" for full trade access (default), "accounts" for view-only'
    )

    args = parser.parse_args()

    # Get credentials from environment variables
    client_id = os.getenv("CTRADER_CLIENT_ID")
    client_secret = os.getenv("CTRADER_CLIENT_SECRET")
    redirect_uri = args.redirect_uri or os.getenv("CTRADER_REDIRECT_URI", "http://localhost:8000/callback")

    # Validate required environment variables
    if not client_id:
        print("Error: CTRADER_CLIENT_ID environment variable is not set")
        print("\nPlease set it with:")
        print("  export CTRADER_CLIENT_ID='your_client_id'")
        sys.exit(1)

    if not client_secret:
        print("Error: CTRADER_CLIENT_SECRET environment variable is not set")
        print("\nPlease set it with:")
        print("  export CTRADER_CLIENT_SECRET='your_client_secret'")
        sys.exit(1)

    scope = args.scope

    print("=" * 70)
    print("cTrader Open API - Access Token Generator")
    print("=" * 70)
    print(f"\nClient ID: {client_id[:10]}...{client_id[-4:] if len(client_id) > 14 else client_id}")
    print(f"Redirect URI: {redirect_uri}")
    print(f"Scope: {scope} {'(TRADE permissions)' if scope == 'trading' else '(VIEW-ONLY - cannot place orders!)'}")
    print()

    # Generate authorization URL
    auth_url = get_authorization_url(client_id, redirect_uri, scope=scope)

    # Determine mode
    if args.manual:
        mode = 'manual'
    elif args.interactive:
        mode = 'interactive'
    else:
        # Ask user which mode they prefer
        print("Choose a mode:")
        print("1. Interactive (recommended) - starts a local server to receive the code automatically")
        print("2. Manual - you'll paste the authorization code yourself")
        choice = input("\nEnter choice (1 or 2): ").strip()
        mode = 'interactive' if choice == '1' else 'manual'

    if mode == 'interactive':
        # Extract port from redirect URI if it's a localhost URL
        parsed_uri = urlparse(redirect_uri)
        if parsed_uri.hostname == 'localhost' and parsed_uri.port:
            port = parsed_uri.port
        else:
            port = args.port
            # Update redirect URI to use the port
            redirect_uri = f"http://localhost:{port}/callback"

        print(f"\nStarting callback server on http://localhost:{port}/callback")
        print("Waiting for authorization...")

        authorization_code_queue = start_callback_server(port, redirect_uri)

        print(f"\nPlease open this URL in your browser:")
        print(f"\n{auth_url}\n")
        print("After authorizing, the script will automatically receive the code.")
        print("Waiting for callback... (Press Ctrl+C to cancel)")

        # Wait for authorization code
        max_wait = 300  # 5 minutes
        waited = 0
        while not authorization_code_queue and waited < max_wait:
            time.sleep(1)
            waited += 1
            if waited % 10 == 0:
                print(f"Still waiting... ({waited}s)")

        if not authorization_code_queue:
            print("\nTimeout: No authorization code received.")
            print("Please try again or use --manual mode.")
            sys.exit(1)

        authorization_code = authorization_code_queue[0]
        print(f"\n✓ Authorization code received!")

    else:  # manual mode
        print(f"\nPlease open this URL in your browser:")
        print(f"\n{auth_url}\n")
        print("After authorizing, you'll be redirected to your redirect URI.")
        print("Copy the 'code' parameter from the URL and paste it below.")
        print("\nExample URL:")
        print("  http://localhost:8000/callback?code=ABC123XYZ...")
        print("\nEnter the authorization code:")
        authorization_code = input("> ").strip()

        if not authorization_code:
            print("Error: No authorization code provided")
            sys.exit(1)

    # Exchange code for token
    print("\nExchanging authorization code for access token...")
    try:
        token_data = exchange_code_for_token(
            authorization_code,
            client_id,
            client_secret,
            redirect_uri
        )

        access_token = token_data.get('accessToken')
        if not access_token:
            print("Error: No access token in response")
            print(f"Response: {token_data}")
            sys.exit(1)

        print("\n" + "=" * 70)
        print(f"SUCCESS! Access Token obtained (scope: {scope})")
        print("=" * 70)
        print(f"\nAccess Token: {access_token}")
        print(f"Scope: {scope} {'- can place trades' if scope == 'trading' else '- VIEW ONLY, cannot trade!'}\n")
        print("Add this to your environment variables:")
        print(f"  export CTRADER_ACCESS_TOKEN='{access_token}'")
        print("\nOr add it to your .env file:")
        print(f"  CTRADER_ACCESS_TOKEN={access_token}")
        print()

        # Optionally save to .env file
        save_to_env = input("Save to .env file? (y/n): ").strip().lower()
        if save_to_env == 'y':
            env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            with open(env_file, 'a') as f:
                f.write(f"\n# cTrader Access Token (generated by get_ctrader_token.py)\n")
                f.write(f"CTRADER_ACCESS_TOKEN={access_token}\n")
            print(f"\n✓ Saved to {env_file}")

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

