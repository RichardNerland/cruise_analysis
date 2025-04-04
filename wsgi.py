from cruise_app import server

# This file makes it explicit that the server should be imported from cruise_app.py
# This helps Render find the correct WSGI application

if __name__ == "__main__":
    server.run() 