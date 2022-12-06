import tkinter as tk

from PIL import ImageGrab

from Accuracies import *
from Activations import *
from Layers import *
from Losses import *
from Optimisers import *
from Model import *
from Datahandler import *

THICK_BRUSH_THICKNESS = 35
THIN_BRUSH_THICKNESS = 10

def submit_drawing():
    ImageGrab.grab().crop((15, 115, 1205, 1330)).save('submitted_image.png')
    
    image_data = load_personal()
    image_data = cv2.resize(image_data, (28, 28))
    image_data = MAX_PIXEL_VALUE - image_data
    plt.imshow(image_data, cmap='gray')
    plt.show()
    image_data = scale_data(image_data)

    predictions = model.predict(image_data)
    predictions = model.output_layer_activation.predictions(predictions)

    prediction = predictions[0]
    resultLabel.config(text=prediction)

def clear_canvas():
    canvas.delete('all')

def paint_oval(event):
    canvas.create_oval(event.x-THIN_BRUSH_THICKNESS, event.y-THIN_BRUSH_THICKNESS, event.x+THIN_BRUSH_THICKNESS, event.y+THIN_BRUSH_THICKNESS, fill='black', outline='black')
    canvas.create_oval(event.x-THICK_BRUSH_THICKNESS, event.y-THICK_BRUSH_THICKNESS, event.x+THICK_BRUSH_THICKNESS, event.y+THICK_BRUSH_THICKNESS, fill='black', outline='black')

def create_window():
    global root, canvas, resultLabel, submitButton, clearButton
    root = tk.Tk()
    root.title("Rory's Handwritten Digit Predictor")
    root.geometry('600x700')

    canvas = tk.Canvas(root, width=600, height=600, bg='white')
    canvas.bind('<B1-Motion>', paint_oval)
    canvas.pack()

    submitButton = tk.Button(root, text='Submit Drawing', fg='black', command=submit_drawing)
    submitButton.pack()

    resultLabel = tk.Label(root, text='Nothing yet...')
    resultLabel.pack()

    clearButton = tk.Button(root, text='Clear drawing', fg='black', command=clear_canvas)
    clearButton.pack()

    root.mainloop()

def create_and_save_model():
    print('Creating model...')
    model = Model()

    model.add(Layer_Dense(X.shape[1], 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dropout(0.2))
    model.add(Layer_Dense(128, 128))
    model.add(Activation_ReLU())
    model.add(Layer_Dense(128, 10))
    model.add(Activation_Softmax())

    model.set(
        loss=Loss_CategoricalCrossEntropy(),
        optimiser=Optimiser_Adam(learning_rate=0.001, decay=1e-3),
        accuracy=Accuracy_Categorical(),
    )
    model.finalise()

    print('Beginning training...')
    model.train(X, y, epochs=10, display_interval=100, validation_data=(X_test, y_test), batch_size=100)
    model.evaluate(X_test, y_test, batch_size=100)
    model.saveModel('bestModel')

#X, y = load_data('train')
X_test, y_test = load_data('test')
#create_and_save_model()
model = Model.loadModel('bestModel')
model.evaluate(X_test, y_test, batch_size=100)
create_window()