import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

# Create a PyQt application
app = QApplication(sys.argv)

# Create a main window
window = QWidget()
window.setWindowTitle("Simple PyQt App")
window.setGeometry(100, 100, 400, 200)  # Set window position (x, y) and size (width, height)

# Create a label
label = QLabel("Hello, PyQt!")
label.move(150, 80)  # Set label position (x, y) within the window

# Show the window
window.show()

# Run the application's event loop
sys.exit(app.exec_())
