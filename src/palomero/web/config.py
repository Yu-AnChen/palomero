import pathlib
import shutil


# Define the application's root directory in the user's home folder
APP_DIR = pathlib.Path.home() / ".palomero"
APP_DIR.mkdir(parents=True, exist_ok=True)

# Define paths for database and task assets
DB_PATH = APP_DIR / "palomero.db"

PUBLIC_DIR = APP_DIR / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

# Copy custom CSS to public dir
curr = pathlib.Path(__file__).resolve().parent
shutil.copytree(curr / "public", PUBLIC_DIR, dirs_exist_ok=True, symlinks=False)
