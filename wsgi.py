from cruise_app import app, server

# This file provides the correct WSGI application for Render deployment
# Exporting both app and server to cover all bases

if __name__ == "__main__":
    app.run_server() 