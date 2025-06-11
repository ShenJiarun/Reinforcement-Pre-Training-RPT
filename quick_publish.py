#!/usr/bin/env python3
"""
Quick publish script for RPT package.
This automatically handles the PyPI publishing process.
"""

import subprocess
import sys
import os
import time

def run_cmd(cmd):
    """Run command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return None

def main():
    print("🚀 Quick Publishing RPT Package to PyPI")
    print("=" * 50)
    
    # Step 1: Clean and build
    print("\n📦 Step 1: Building package...")
    
    # Clean
    if run_cmd("rm -rf dist/ build/ *.egg-info/") is None:
        return False
    
    # Build
    if run_cmd("python3 -m build") is None:
        print("❌ Failed to build package")
        return False
    
    # Check
    if run_cmd("python3 -m twine check dist/*") is None:
        print("❌ Package check failed")
        return False
    
    print("✅ Package built successfully!")
    
    # Step 2: Show what will be uploaded
    print("\n📋 Package contents:")
    run_cmd("ls -la dist/")
    
    # Step 3: Upload instructions
    print("\n🔑 To complete the upload, you need:")
    print("1. A PyPI account: https://pypi.org/account/register/")
    print("2. An API token: https://pypi.org/manage/account/token/")
    print("\n📤 Then run this command:")
    print("python3 -m twine upload dist/* --username __token__ --password YOUR_TOKEN_HERE")
    
    print("\n🎯 Or use the interactive script:")
    print("python3 publish_to_pypi.py")
    
    print("\n✨ Package is ready for publishing!")
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Build completed successfully!")
        
        # Try one more automated upload attempt
        print("\n🚀 Attempting automated upload...")
        
        # This requires the user to have PYPI_TOKEN environment variable set
        token = os.environ.get('PYPI_TOKEN')
        if token:
            print("Found PYPI_TOKEN environment variable, uploading...")
            result = subprocess.run(
                f'python3 -m twine upload dist/* --username __token__ --password {token}',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print("🎉 Successfully uploaded to PyPI!")
                print("📦 Install with: pip install reinforcement-pretraining")
                print("🔗 View at: https://pypi.org/project/reinforcement-pretraining/")
            else:
                print(f"❌ Upload failed: {result.stderr}")
                print("💡 Try running: python3 publish_to_pypi.py")
        else:
            print("💡 Set PYPI_TOKEN environment variable for automatic upload")
            print("💡 Or run: python3 publish_to_pypi.py")
    else:
        print("❌ Build failed!")
        sys.exit(1)