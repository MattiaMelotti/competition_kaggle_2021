import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom          #type: ignore
from tensorflow.keras import datasets, layers, models                               #type: ignore
from tensorflow.keras.models import load_model                                      #type: ignore
from tensorflow.keras.optimizers import Adam                                        #type: ignore

def addestra_modello():
    #prima di iniziare qualsiasi esercizio, è SEMPRE FONDAMENTALE vedere che tipo di dati ci sono in OGNI file
    train_data = pd.read_csv('training.csv')
    test_data = pd.read_csv('testing.csv')
    #ogni numero all'interno di ogni riga mi rappresenta l'intensità di colore di ogni pixel dell'immagine
    #ogni riga ha 96x96 valori, una stringa di 9216

    #guardo le caratteristiche dei due set di dati
    # print(train_data.shape)
    # print(test_data.shape)
    # print(train_data.info())
    # print(test_data.info())

    #ci sono un sacco di celle vuote, devo applicare la media, moda o mediama per riempirli oppure eliminarle
    data_complete = train_data.dropna()
    # print(data_complete.info())

    #creo una funzione per prendere l'immagine e trasformarla in un array
    def img_from_csv(dataframe):
        img_list = []
        for i in dataframe['Image']:
            pix = np.array([int(p) for p in i.split()], dtype=np.float64)
            img = pix.reshape(96,96,1)                                          #posso non riportare il '-1' iniziale perchè tanto ciclo tutti gli argomenti
            img_list.append(img)
        return np.stack(img_list, axis=0)/255.0                                 #stack è utile quanto devo mettere insieme più array, restituendomi una matrice

    #divido il target dal training. E' al contrario perchè sto facendo una regressione (immagine -> dati)
    y = data_complete.drop(columns='Image')
    y = y.values.astype(np.float64)/96.0
    X = img_from_csv(data_complete)

    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.3,random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # print(X_val.shape,X_test.shape) #per vedere quanti sono

    #####questo è un altro modo per creare la y_train########
    #creo una directory per caricarci le immagini
    os.makedirs('immagini_trasformate',exist_ok=True)

    # #creo un ciclo per exportare tutte le immagini del dataset
    # for i in range(5000):
    #     image_str = train_data.loc[i,'Image']                                   #estraiamo la stringa dell' immagine (posizione i)
    #     pixels = np.array([int(p) for p in image_str.split()],dtype=np.uint8)   #
    #     pixels = pixels.reshape(96,96)
    #     immagine = Image.fromarray(pixels,mode='L')                             #li estraiamo in scala di grigi
    #     immagine.save(r'C:\Users\matti\Desktop\corsi_di_formazione\2025_02_BitCamp_python_AI\esercizi\esercizi_reti_neurali\tjml-2021-2022-cnn-competition_kaggle\immagini_trasformate'+f'\immagine_{i}.jpg')

    # data_augmentation = tf.keras.Sequential([
    #     RandomFlip('horizontal'),
    #     RandomRotation(0.1),
    #     RandomZoom(0.1),
    # ], name='data_augmentation')

    # Crea il modello CNN secondo le indicaziioni dell'esercizio impostando il numero di neuroni più adeguato
    modello = models.Sequential([
        layers.Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(96,96,1)),             #dichiaro anche la forma, va fatto solo nel primo strato convoluzionale
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),                                    #il padding qua è essenziale perchè ...
        layers.MaxPooling2D((2,2)),                                                                     # Primo strato di pooling
        layers.Dropout(0.2),                                                                            #il dorpuot bilancia l'apprendimento su tutti i neuroni (non utilizza sempre gli stessi)

        #Il blocco Conv–Conv–Pool è uno standard perché:
        # - Estende il campo recettivo senza aumentare troppo i parametri.
        # - Estrae più informazione a pari risoluzione.
        # - Migliora la capacità di generalizzazione.
        # - È molto più stabile che una singola Conv grossa seguita subito da Pool.

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),  
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(256, (3,3), padding='same', activation='relu'),
        layers.Conv2D(256, (3,3), padding='same', activation='relu'),  
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(30, activation='sigmoid')  
        #meglio sigmoid perchè mi restituisce valori fra 0 e 1. ideale per regressione continua o classificazione binaria
        #softmax è meglio utilizzarlo per una classificazione multiclasse
    ])

    # Compilo il modello
    modello.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse', #perchè andiamo a fare la regressione
        metrics=["mae","mse"]  #min absolute error e poi l'errore quadratico medio. facendo regressione, non ha senso trovare l'accuratezza
    )

    #addestro il modello inserendo le lineee guida richieste
    # history = modello.fit(X,y,epochs=50,batch_size=16,verbose=1)

    #ho fatto questa modifica all'addestramento perchè credo sia più corretto allenarlo sul train e valutarlo sulla parte di test
    history = modello.fit(X_train,y_train, validation_data=(X_val,y_val), epochs=50,batch_size=16,verbose=1)
    modello.evaluate(X_test,y_test,verbose=0)

    modello.save('face_model.h5')

    print(f'Training metrics per epoch: {history.history["mae"]}, {history.history["mse"]}')

from tkinter import Tk, Button, Label, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw

img_arr = None
img_png = None

def gui():
    root = Tk()
    root.geometry('500x500+300+200')
    root.title('Riconoscimento facciale')
    root.minsize(600,400)
    root.columnconfigure(0,weight=1)
    root.columnconfigure(1,weight=1)

    def load_img():
        path = filedialog.askopenfilename(title='Selezione un\'immagine',filetypes=[('immagini','*.png *.jpg *.jpeg')])
        global img_arr, img_png

        if not path:
            messagebox.showerror('ERRORE','Percorso del file non trovato')
        
        try:
            img_clean = Image.open(path).convert('L')
            img_clean = img_clean.resize((96,96),Image.LANCZOS)
            img_png = img_clean  #assegno l'immagine pulita alla variabile globale
            img_clean_tk = img_clean.resize((96,96), Image.LANCZOS)
            root.image_clean_tk = ImageTk.PhotoImage(img_clean_tk)

            arr = np.array(img_clean,dtype=np.float64)/255.0
            arr = arr.reshape(1,96,96,1)
            img_arr = arr

            lbl_img_clean.config(text=f'Image uploaded: {os.path.basename(path)}',image=root.image_clean_tk,compound='bottom')
            lbl_load.config(text='Image selected')
            lbl_recogn.config(text='Ready to recognize')
            lbl_img_mod.config(text='')
            root.img_draw = None

        except Exception as e:
            messagebox.showerror(title='Errore',message=e)
            lbl_load.config(text='Image path not found')
            lbl_img_mod.config(text='')
            root.img_draw = None   

    def recognize():
        model = load_model('face_model.h5',compile=False)
        global img_arr, img_png

        if img_arr is None:
            messagebox.showwarning(title='WARNING',message='Image not found')

        pred = model.predict(img_arr,verbose=0)[0]

        # 2) Coppie (15,2) in [0,1]
        pts01 = pred.reshape(-1, 2)

        # 3) Scegli su quale immagine disegnare (qui: quella che mostri a schermo)
        img_draw = img_png.copy()
        img_draw = img_draw.convert('RGB')
        Wt, Ht = img_draw.size  # target width/height della PIL di disegno

        # 4) Scala alle dimensioni di disegno (niente reshape a 96: uso direttamente Wt,Ht)
        pts_pixel = np.column_stack([
            np.clip(np.rint(pts01[:, 0] * Wt), 0, Wt - 1),
            np.clip(np.rint(pts01[:, 1] * Ht), 0, Ht - 1)
        ]).astype(int)

        # 5) Indici dal tuo CSV
        EYES  = [0, 1, 2, 3, 4, 5]       # left/right eye center + inner/outer corners
        NOSE  = [10]                     # nose_tip
        MOUTH = [11, 12, 13, 14]         # mouth left/right + top/bottom lip
        to_plot = EYES + NOSE + MOUTH

        # 6) Disegna puntini rossi
        draw = ImageDraw.Draw(img_draw)                 # 1) contesto di disegno legato a 'img_draw'
        r = max(2, int(0.01 * max(Wt, Ht)))             # 2) raggio: 1% del lato maggiore, almeno 2 px
        for idx in to_plot:                             # 3) scorri i landmarks selezionati
            x, y = pts_pixel[idx]                       #    coordinate pixel (int) già scalate all'immagine
            draw.ellipse((x-r, y-r, x+r, y+r),          # 4) cerchietto centrato sul punto
                        fill='red', outline='red', width=1)
        root.img_draw = ImageTk.PhotoImage(img_draw)

        lbl_img_mod.config(text='Image with facial points',image=root.img_draw,compound='bottom')
        lbl_recogn.config(text='Recognition successful')


    btn_load = Button(root,text='Load image',borderwidth=2,width=40,command=load_img,pady=10)
    btn_load.grid(column=0,row=0,pady=20)

    lbl_load = Label(root, text='No images selected')
    lbl_load.grid(column=0,row=1)

    btn_recogn = Button(root,text='Facial recognition',width=40,pady=10,command=recognize)
    btn_recogn.grid(column=0,row=2,pady=20)

    lbl_recogn = Label(root,text='Still no recognition possible')
    lbl_recogn.grid(column=0,row=3)

    lbl_img_clean = Label(root)
    lbl_img_clean.grid(column=1,row=0,rowspan=2,pady=20)

    lbl_img_mod = Label(root)
    lbl_img_mod.grid(column=1,row=2,rowspan=2,pady=20)

    root.mainloop()