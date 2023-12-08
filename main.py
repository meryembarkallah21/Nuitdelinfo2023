import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import tkinter as tk
from PIL import Image, ImageTk


def train_models():
    # Load the dataset
    data = pd.read_csv('falseortrue.csv')

    # Split data into features and labels
    X = data['v2']
    y = data['v1']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    # Initialize different models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Random Forest": RandomForestClassifier(),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Support Vector Machine": SVC()
    }

    best_model = None
    best_accuracy = 0.0
    for name, model in models.items():
        model.fit(X_train_vect, y_train)
        accuracy = model.score(X_test_vect, y_test)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model, vectorizer

def check_text():
    text = entry.get()
    text_vectorized = vectorizer.transform([text])
    
    prediction = best_model.predict(text_vectorized)
    result_label.config(text=f"Prediction: {prediction[0]} c:")

# Train the models and get the best model
best_model, vectorizer = train_models()

""" # Create Tkinter window
root = tk.Tk()
root.title("")

# Entry widget to input text
entry = tk.Entry(root, width=50)
entry.pack()

# Button to trigger prediction
check_button = tk.Button(root, text="Check", command=check_text)
check_button.pack()

# Label to display the prediction result
result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
 """

def on_entry_click(event):
    if entry.get() == 'Type here':
       entry.delete(0, "end")
       entry.config(fg='black')

def on_focusout(event):
    if entry.get() == '':
        entry.insert(0, 'Type here')
        entry.config(fg='grey')

def set_border(widget, width):
    widget.config(
        highlightbackground="black",
        highlightcolor="black",
        highlightthickness=width,
        bd=0
    )



root = tk.Tk()
root.title("Nuit de l'info c:")

# Set window size and position it at the center of the screen
window_width = 1100
window_height = 950
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Load background image
bg_image = tk.PhotoImage(file="1.png")
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

# Title Label
title_label = tk.Label(root, text="TOPIC 2023: DEALING WITH CLIMATE CHANGE: THE TRUE FROM THE FALSE", font=("Arial", 14, "bold"), bg="grey")
title_label.pack(pady=10)

# Entry widget to input text
""" entry = tk.Entry(root, width=50, font=("Arial", 30))
entry.pack() """

entry = tk.Entry(root, width=50, font=("Arial", 30))
entry.insert(0, 'Type here')
entry.config(fg='grey')
entry.bind('<FocusIn>', on_entry_click)
entry.bind('<FocusOut>', on_focusout)
entry.pack()

set_border(entry, 2)  

""" # Button with image
button_image = tk.PhotoImage(file="op.png")
check_button = tk.Button(root, image=button_image, command=check_text, bd=0)
check_button.pack(pady=8)
 """



# Button with image
button_image = tk.PhotoImage(file="op.png")
# Resize the button image to fit within a smaller size
small_button_image = button_image.subsample(6, 6)  # Change the numbers to adjust the size
check_button = tk.Button(root, image=small_button_image, command=check_text, bd=0)
check_button.pack(pady=8)


# Label to display the prediction result
result_label = tk.Label(root, text="", font=("Arial", 12), bg="white")
result_label.pack()

# Examples at the bottom
examples_frame = tk.Frame(root, bg="white")
examples_frame.pack(pady=10)

example_text = [
    "e.g:",
    "Factories' carbon emissions have an effect on air quality",
    "Trees aid in combating climate change by absorbing CO2.",
    "Switching to clean energy sources has no impact on reducing emissions.",
    "Climate change is a result of natural Earth cycles.",
    "Human activities have no influence on the ozone layer."
]

for text in example_text:
    example_label = tk.Label(examples_frame, text=text, font=("Arial", 10), fg="grey", wraplength=window_width-20, anchor="w", justify="left")
    example_label.pack(anchor="w", padx=10, pady=2)

root.mainloop()
