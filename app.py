# This file is needed because Render falls back to trying to import 'app' when it can't find the specified module
# Simply re-export the app and server from cruise_app.py

from cruise_app import app, server

# Export both app and server for flexibility
# This handles Render's fallback to 'app:app'

if __name__ == "__main__":
    app.run_server() 