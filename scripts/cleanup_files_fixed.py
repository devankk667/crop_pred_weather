import os
import shutil

def cleanup_files():
    """Remove unnecessary files and keep only essential ones."""
    # Files to keep
    essential_files = {
        'final_crop_soil_data_with_yield_fixed.csv',  # Final processed data
        'final_soil_data_2015_2020.csv',             # Soil data
        'crop-wise-area-production-yield.csv',        # Original data
        'solidata.csv',                              # Original soil data
        'nasa_weather.py',                           # Weather data script
        'district_coordinates.csv',                   # District coordinates
        'fetch_weather_data.py',                     # Weather data fetching script
        'weather_data/seasonal/all_seasonal_data.csv',  # Weather data
        'cleanup_files_fixed.py',                    # This script
    }
    
    # Get all files in the current directory
    all_files = os.listdir('.')
    
    # Find files to remove (everything not in essential_files)
    files_to_remove = [f for f in all_files if f not in essential_files and 
                      (f.endswith('.py') or f.endswith('.csv~') or f.endswith('.py~') or 
                       f == '__pycache__' or f.endswith('.pyc'))]
    
    if not files_to_remove:
        print("No files to remove.")
        return
    
    # Show files to be removed
    print("The following files will be removed:")
    for f in sorted(files_to_remove):
        print(f"- {f}")
    
    # Ask for confirmation
    confirm = input("\nAre you sure you want to remove these files? (y/n): ")
    if confirm.lower() == 'y':
        # Remove files
        for f in files_to_remove:
            try:
                if os.path.isfile(f):
                    os.remove(f)
                    print(f"Removed: {f}")
                elif os.path.isdir(f):
                    shutil.rmtree(f)
                    print(f"Removed directory: {f}")
            except Exception as e:
                print(f"Error removing {f}: {e}")
        print("\nCleanup complete!")
    else:
        print("\nCleanup cancelled.")

if __name__ == "__main__":
    cleanup_files()
