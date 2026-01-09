#!/usr/bin/env python3
"""
Quick Image Viewer for Stunting Classification Results
Run this script to view generated figures without re-running the pipeline.

Supports inline image display in Warp/iTerm2 terminals!
"""

import subprocess
import sys
from pathlib import Path

FIGURES_PATH = Path(__file__).parent / "reports" / "figures"


def display_image_imgcat(image_path):
    """Display image using imgcat library (best for iTerm2/Warp)."""
    import time
    import sys
    try:
        from imgcat import imgcat
        
        # Flush before displaying
        sys.stdout.flush()
        
        print(f"\n  📈 {image_path.name}")
        print("  " + "─"*60)
        sys.stdout.flush()
        
        # Small delay to let terminal catch up
        time.sleep(0.3)
        
        with open(image_path, 'rb') as f:
            imgcat(f.read())
        
        sys.stdout.flush()
        time.sleep(0.2)  # Delay after image
        print()
        return True
    except Exception as e:
        print(f"  Error with imgcat: {e}")
        return False


def display_image_rich(image_path):
    """Display image info using rich (fallback)."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
        from PIL import Image
        
        console = Console()
        
        # Get image info
        img = Image.open(image_path)
        width, height = img.size
        
        console.print(Panel(
            f"[bold cyan]{image_path.name}[/bold cyan]\n"
            f"Size: {width} x {height} pixels\n"
            f"Path: {image_path}",
            title="🖼️ Image",
            border_style="green"
        ))
        return True
    except Exception as e:
        print(f"  Error with rich: {e}")
        return False


def display_images_inline(image_files, one_by_one=False):
    """Display multiple images inline in terminal using imgcat."""
    import sys
    import time
    
    print("\n" + "═"*70)
    print(f"  📊 INLINE IMAGE DISPLAY ({len(image_files)} images)")
    print("═"*70)
    sys.stdout.flush()
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n  [{i}/{len(image_files)}]", end="")
        sys.stdout.flush()
        
        if not display_image_imgcat(img_path):
            # Fallback to rich if imgcat fails
            display_image_rich(img_path)
        
        # Pause between images if requested
        if one_by_one and i < len(image_files):
            try:
                input("  Press Enter for next image...")
            except EOFError:
                pass


def list_images():
    """List all available images."""
    image_files = sorted(FIGURES_PATH.glob("*.png"))
    
    if not image_files:
        print("❌ No images found in reports/figures/")
        print("   Run main.py first to generate visualizations.")
        return []
    
    print("\n" + "═"*60)
    print("  🖼️  AVAILABLE FIGURES")
    print("═"*60 + "\n")
    
    categories = {
        'confusion': [],
        'roc': [],
        'distribution': [],
        'feature': [],
        'comparison': [],
        'other': []
    }
    
    for img in image_files:
        name = img.name.lower()
        if 'confusion' in name:
            categories['confusion'].append(img)
        elif 'roc' in name:
            categories['roc'].append(img)
        elif 'distribution' in name or 'target' in name:
            categories['distribution'].append(img)
        elif 'feature' in name or 'importance' in name:
            categories['feature'].append(img)
        elif 'comparison' in name or 'metric' in name or 'cv_' in name:
            categories['comparison'].append(img)
        else:
            categories['other'].append(img)
    
    idx = 1
    image_map = {}
    
    cat_labels = {
        'confusion': '📊 Confusion Matrices',
        'roc': '📈 ROC Curves', 
        'distribution': '📉 Data Distributions',
        'feature': '🔍 Feature Analysis',
        'comparison': '🏆 Model Comparison',
        'other': '📁 Other'
    }
    
    for cat, imgs in categories.items():
        if imgs:
            print(f"  {cat_labels[cat]}:")
            for img in imgs:
                print(f"    [{idx:2}] {img.name}")
                image_map[idx] = img
                idx += 1
            print()
    
    return image_map

def main():
    """Main interactive viewer."""
    while True:
        image_map = list_images()
        
        if not image_map:
            return
        
        print("  Commands:")
        print("    [A]  Open ALL images")
        print("    [C]  Open Confusion Matrices")
        print("    [R]  Open ROC Curves")
        print("    [D]  Open Distributions")
        print("    [F]  Open Feature Analysis")
        print("    [M]  Open Model Comparison")
        print("    [1-99] Open specific image")
        print("    [Q]  Quit")
        print()
        
        try:
            choice = input("  Your choice: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break
        
        if choice == 'Q':
            print("  Goodbye!")
            break
        elif choice == 'A':
            print("\n  Opening all images...")
            subprocess.run(['open'] + [str(f) for f in image_map.values()], check=False)
        elif choice == 'C':
            files = [f for f in image_map.values() if 'confusion' in f.name.lower()]
            if files:
                subprocess.run(['open'] + [str(f) for f in files], check=False)
        elif choice == 'R':
            files = [f for f in image_map.values() if 'roc' in f.name.lower()]
            if files:
                subprocess.run(['open'] + [str(f) for f in files], check=False)
        elif choice == 'D':
            files = [f for f in image_map.values() if 'distribution' in f.name.lower() or 'target' in f.name.lower()]
            if files:
                subprocess.run(['open'] + [str(f) for f in files], check=False)
        elif choice == 'F':
            files = [f for f in image_map.values() if 'feature' in f.name.lower() or 'importance' in f.name.lower()]
            if files:
                subprocess.run(['open'] + [str(f) for f in files], check=False)
        elif choice == 'M':
            files = [f for f in image_map.values() if 'comparison' in f.name.lower() or 'metric' in f.name.lower() or 'cv_' in f.name.lower()]
            if files:
                subprocess.run(['open'] + [str(f) for f in files], check=False)
        elif choice.isdigit():
            idx = int(choice)
            if idx in image_map:
                print(f"\n  Opening {image_map[idx].name}...")
                subprocess.run(['open', str(image_map[idx])], check=False)
            else:
                print("  ❌ Invalid number")
        else:
            print("  ❌ Invalid choice")
        
        print("\n" + "-"*60 + "\n")


if __name__ == "__main__":
    image_files = sorted(FIGURES_PATH.glob("*.png"))
    
    # Check for slow mode flag
    slow_mode = '--slow' in sys.argv or '-s' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    
    if args:
        arg = args[0].lower()
        
        if arg == 'all':
            display_images_inline(image_files, one_by_one=slow_mode)
        elif arg == 'confusion':
            files = [f for f in image_files if 'confusion' in f.name.lower()]
            display_images_inline(files, one_by_one=slow_mode)
        elif arg == 'roc':
            files = [f for f in image_files if 'roc' in f.name.lower()]
            display_images_inline(files, one_by_one=slow_mode)
        elif arg == 'distribution':
            files = [f for f in image_files if 'distribution' in f.name.lower() or 'target' in f.name.lower()]
            display_images_inline(files, one_by_one=slow_mode)
        elif arg == 'feature':
            files = [f for f in image_files if 'feature' in f.name.lower() or 'importance' in f.name.lower()]
            display_images_inline(files, one_by_one=slow_mode)
        elif arg == 'comparison':
            files = [f for f in image_files if 'comparison' in f.name.lower() or 'metric' in f.name.lower() or 'cv_' in f.name.lower()]
            display_images_inline(files, one_by_one=slow_mode)
        elif arg == 'open':
            subprocess.run(['open'] + [str(f) for f in image_files], check=False)
        elif arg == 'list':
            # Just list files
            print("\n🖼️  Available images:")
            for i, f in enumerate(image_files, 1):
                print(f"  {i:2}. {f.name}")
        elif arg.isdigit():
            idx = int(arg) - 1
            if 0 <= idx < len(image_files):
                display_images_inline([image_files[idx]])
            else:
                print(f"  Invalid index. Valid: 1-{len(image_files)}")
        else:
            print("Usage: python view_images.py [COMMAND] [--slow]")
            print("\nCommands:")
            print("  all          - Display all images inline")
            print("  confusion    - Display confusion matrices")
            print("  roc          - Display ROC curves")
            print("  distribution - Display data distributions")
            print("  feature      - Display feature importance")
            print("  comparison   - Display model comparison")
            print("  open         - Open in Preview app instead")
            print("  list         - List all available images")
            print("  <number>     - Display specific image (e.g., 1, 2, 3)")
            print("\nFlags:")
            print("  --slow, -s   - Wait for Enter between images")
            print("\nExamples:")
            print("  python view_images.py confusion")
            print("  python view_images.py all --slow")
            print("  python view_images.py 5")
    else:
        # Default: show all inline
        print("\n🖼️  Displaying all figures inline in terminal...")
        print("    Tip: Use --slow flag if images don't show properly\n")
        display_images_inline(image_files)
