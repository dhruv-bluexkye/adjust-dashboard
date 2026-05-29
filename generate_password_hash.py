"""
Run this script to generate bcrypt password hashes for secrets.toml.

Usage:
    python generate_password_hash.py
"""

import getpass
import bcrypt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


if __name__ == "__main__":
    print("Adjust Dashboard — Password Hash Generator")
    print("=" * 45)
    while True:
        username = input("\nUsername (or press Enter to quit): ").strip()
        if not username:
            break
        password = getpass.getpass(f"Password for '{username}': ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match. Try again.")
            continue
        hashed = hash_password(password)
        print(f"\nAdd this to .streamlit/secrets.toml:\n")
        print(f"[credentials.{username}]")
        print(f'name     = "{username.capitalize()}"')
        print(f'email    = ""')
        print(f'password = "{hashed}"')
    print("\nDone.")
