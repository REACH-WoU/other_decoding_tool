from shiny import App
from www.src.ui import app_ui
from www.src.server import server

# Create shiny application using app_ui object from the www.src.ui.py and server function from the www.src.server.py
app = App(app_ui, server)
app.run()
