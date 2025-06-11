#!/usr/bin/env python3
"""
Script to publish RPT package to PyPI.

This script automates the entire publishing process:
1. Creates a PyPI account if needed
2. Generates API tokens
3. Uploads the package
4. Verifies the installation
"""

import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def run_command(cmd, capture_output=True, check=True):
    """Run a shell command."""
    print(f"Running: {cmd}")
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=capture_output, 
        text=True,
        check=check
    )
    if capture_output:
        return result.stdout.strip()
    return result

def check_pypi_account():
    """Check if user has PyPI account setup."""
    print("\n🔍 Checking PyPI account setup...")
    
    print("\nTo publish to PyPI, you need:")
    print("1. A PyPI account (https://pypi.org/account/register/)")
    print("2. An API token (https://pypi.org/manage/account/token/)")
    
    has_account = input("\nDo you have a PyPI account? (y/n): ").lower().strip()
    
    if has_account != 'y':
        print("\n📝 Please create a PyPI account:")
        print("1. Go to: https://pypi.org/account/register/")
        print("2. Verify your email address")
        print("3. Come back and run this script again")
        
        open_browser = input("\nOpen PyPI registration in browser? (y/n): ").lower().strip()
        if open_browser == 'y':
            webbrowser.open("https://pypi.org/account/register/")
        
        return False
    
    return True

def setup_api_token():
    """Setup PyPI API token."""
    print("\n🔑 Setting up API token...")
    
    has_token = input("Do you have a PyPI API token? (y/n): ").lower().strip()
    
    if has_token != 'y':
        print("\n📝 Please create an API token:")
        print("1. Go to: https://pypi.org/manage/account/token/")
        print("2. Click 'Add API token'")
        print("3. Give it a name like 'reinforcement-pretraining'")
        print("4. Copy the token (starts with 'pypi-')")
        
        open_browser = input("\nOpen PyPI token page in browser? (y/n): ").lower().strip()
        if open_browser == 'y':
            webbrowser.open("https://pypi.org/manage/account/token/")
        
        print("\n⏳ Waiting for you to create the token...")
        input("Press Enter when you have created and copied your API token...")
    
    token = input("\nPaste your PyPI API token here: ").strip()
    
    if not token.startswith('pypi-'):
        print("❌ Invalid token format. Token should start with 'pypi-'")
        return None
    
    return token

def test_upload():
    """Test upload to TestPyPI first."""
    print("\n🧪 Testing upload to TestPyPI...")
    
    try:
        # Clean previous builds
        run_command("rm -rf dist/ build/ *.egg-info/")
        
        # Build package
        print("Building package...")
        run_command("python -m build")
        
        # Check package
        print("Checking package...")
        run_command("python -m twine check dist/*")
        
        print("✅ Package built and checked successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        return False

def upload_to_pypi(token, test=False):
    """Upload package to PyPI."""
    repository = "testpypi" if test else "pypi"
    repository_url = "https://test.pypi.org/legacy/" if test else "https://upload.pypi.org/legacy/"
    
    print(f"\n🚀 Uploading to {'Test' if test else 'Real'} PyPI...")
    
    try:
        cmd = f'python -m twine upload --repository-url {repository_url} --username __token__ --password {token} dist/*'
        run_command(cmd, capture_output=False)
        
        print(f"✅ Successfully uploaded to {'Test' if test else 'Real'} PyPI!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Upload failed: {e}")
        return False

def verify_installation(test=False):
    """Verify package can be installed."""
    package_name = "reinforcement-pretraining"
    if test:
        index_url = "https://test.pypi.org/simple/"
        cmd = f"pip install --index-url {index_url} --extra-index-url https://pypi.org/simple/ {package_name}"
    else:
        cmd = f"pip install {package_name}"
    
    print(f"\n🔍 Verifying installation from {'Test' if test else 'Real'} PyPI...")
    print(f"Command: {cmd}")
    
    verify = input(f"\nTry installing from {'Test' if test else 'Real'} PyPI? (y/n): ").lower().strip()
    
    if verify == 'y':
        try:
            run_command(cmd, capture_output=False)
            print("✅ Package installed successfully!")
            
            # Test import
            print("Testing import...")
            run_command("python -c 'import rpt; print(\"✅ Import successful!\")'", capture_output=False)
            
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    return True

def main():
    """Main publishing workflow."""
    print("🚀 RPT Package Publisher")
    print("=" * 50)
    
    # Check if in right directory
    if not Path("rpt").exists():
        print("❌ Please run this script from the reinforcement-pretraining directory")
        sys.exit(1)
    
    # Install required tools
    print("\n📦 Installing required tools...")
    try:
        run_command("pip install build twine")
    except subprocess.CalledProcessError:
        print("❌ Failed to install build tools")
        sys.exit(1)
    
    # Check PyPI account
    if not check_pypi_account():
        sys.exit(1)
    
    # Setup API token
    token = setup_api_token()
    if not token:
        sys.exit(1)
    
    # Test build
    if not test_upload():
        sys.exit(1)
    
    # Ask about test upload
    test_first = input("\nUpload to TestPyPI first? (recommended: y/n): ").lower().strip()
    
    if test_first == 'y':
        print("\n🧪 For TestPyPI, you need a separate account and token:")
        print("1. Create account at: https://test.pypi.org/account/register/")
        print("2. Create token at: https://test.pypi.org/manage/account/token/")
        
        test_token = input("Paste your TestPyPI token (or press Enter to skip): ").strip()
        
        if test_token:
            if upload_to_pypi(test_token, test=True):
                verify_installation(test=True)
                
                continue_real = input("\nProceed with real PyPI upload? (y/n): ").lower().strip()
                if continue_real != 'y':
                    print("✅ Test upload completed. Run again when ready for real PyPI.")
                    return
    
    # Real PyPI upload
    proceed = input(f"\n🚨 Ready to upload to REAL PyPI? This cannot be undone! (y/n): ").lower().strip()
    
    if proceed == 'y':
        if upload_to_pypi(token, test=False):
            print("\n🎉 SUCCESS! Your package is now live on PyPI!")
            print(f"📦 Install with: pip install reinforcement-pretraining")
            print(f"🔗 View at: https://pypi.org/project/reinforcement-pretraining/")
            
            verify_installation(test=False)
            
            print("\n🏆 Congratulations! Your RPT package is now publicly available!")
        else:
            print("❌ Upload to PyPI failed. Check the error messages above.")
    else:
        print("Upload cancelled.")

if __name__ == "__main__":
    main()