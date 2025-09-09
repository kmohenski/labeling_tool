import tkinter as tk
from src.labeler import Labeler

def main():
    root = tk.Tk()
    app = Labeler(root)
    app.root.mainloop()

if __name__ == "__main__":
    main()