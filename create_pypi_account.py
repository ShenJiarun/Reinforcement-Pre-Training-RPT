#!/usr/bin/env python3
"""
Helper script to guide PyPI account creation and token setup.
"""

import webbrowser
import subprocess
import sys
import time

def main():
    print("🔐 PyPI Account & Token Setup Helper")
    print("=" * 50)
    
    print("\n📝 Step 1: Create PyPI Account")
    print("If you don't have a PyPI account yet:")
    print("1. Go to: https://pypi.org/account/register/")
    print("2. Fill in your details")
    print("3. Verify your email address")
    
    open_register = input("\nOpen PyPI registration page? (y/n): ").lower().strip()
    if open_register == 'y':
        webbrowser.open("https://pypi.org/account/register/")
        print("✅ Opened registration page in browser")
    
    input("\nPress Enter after creating your PyPI account...")
    
    print("\n🔑 Step 2: Generate API Token")
    print("1. Go to: https://pypi.org/manage/account/token/")
    print("2. Click 'Add API token'")
    print("3. Token name: 'reinforcement-pretraining'")
    print("4. Scope: 'Entire account' (for first upload)")
    print("5. Copy the token (starts with 'pypi-')")
    
    open_token = input("\nOpen token creation page? (y/n): ").lower().strip()
    if open_token == 'y':
        webbrowser.open("https://pypi.org/manage/account/token/")
        print("✅ Opened token page in browser")
    
    input("\nPress Enter after creating your API token...")
    
    print("\n🚀 Step 3: Publish Package")
    print("Now you can publish using one of these methods:")
    print("")
    print("Method 1 - Interactive script:")
    print("  python3 publish_to_pypi.py")
    print("")
    print("Method 2 - Manual upload:")
    print("  python3 -m twine upload dist/* --username __token__ --password YOUR_TOKEN")
    print("")
    print("Method 3 - Environment variable:")
    print("  export PYPI_TOKEN='your_token_here'")
    print("  python3 quick_publish.py")
    
    print("\n✅ Setup complete! Your package is ready to publish.")
    
    run_publisher = input("\nRun the interactive publisher now? (y/n): ").lower().strip()
    if run_publisher == 'y':
        try:
            subprocess.run([sys.executable, "publish_to_pypi.py"], check=True)
        except subprocess.CalledProcessError:
            print("💡 Run 'python3 publish_to_pypi.py' manually")
        except FileNotFoundError:
            print("💡 Run 'python3 publish_to_pypi.py' manually")

if __name__ == "__main__":
    main()