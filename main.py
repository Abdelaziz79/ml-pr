from tkinter.filedialog import askdirectory, askopenfilename

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import customtkinter
import tkinter as tk
import csv
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
import seaborn as sns

learning_rate = 0.1
n_neighbors = 5
mse = 0
rmse = 0
mae = 0
csv_path = ''
data_frame = pd.DataFrame()
train_size = 0

x = pd.DataFrame()
y = pd.DataFrame()

x_train, x_test, y_train, y_test = 0, 0, 0, 0

m = LinearRegression()
strategy = ''
columns_name = []
# for imputer
columns_choice = []
# for encoder
columns_encoder = []
encoder = ''
drop_null = ''
tol = 0


def location():
    file = askopenfilename(parent=frame)
    global csv_path
    csv_path = file


# ----------------------------------------------------------------------------------------------------------------------
# root and theme

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("1200x720")
root.title("Machine Learning Project")

# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# frame

frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=20, padx=20, fill="both", expand=True)

frame.columnconfigure(index=0, weight=1)
frame.columnconfigure(index=1, weight=1)

# ----------------------------------------------------------------------------------------------------------------------
# choose file label

main_frame = customtkinter.CTkFrame(frame)
main_frame.grid(row=0, column=0, padx=20, pady=20, sticky=tk.W)
label1 = customtkinter.CTkLabel(main_frame, text='Choose CSV File', font=('Lucida Sans', 24))
label1.grid(row=0, column=0, padx=20, pady=20)


# preprocessing
def pre():
    pre_window = customtkinter.CTkToplevel(frame)
    pre_window.geometry("720x500")
    pre_window.title("preprocessing")
    pre_frame = customtkinter.CTkFrame(master=pre_window)
    pre_frame.pack(pady=20, padx=20, fill="both", expand=True)

    def checkbox_event():
        global encoder
        encoder = check_var.get()

    def checkbox_event1():
        global drop_null
        drop_null = check_var1.get()

    def combo_callback(choice):
        global strategy
        strategy = choice

    def option1_callback(choice):
        global columns_encoder
        columns_encoder.append(choice)
        print(columns_encoder)

    def option_callback(choice):
        global columns_choice
        columns_choice.append(choice)

    def submit1():
        global strategy, columns_choice, columns_encoder, encoder, drop_null, data_frame, x, y
        print(strategy, columns_choice, columns_encoder, encoder, drop_null)
        if strategy != 'none':
            for name in columns_choice:
                imp = SimpleImputer(missing_values=np.nan, strategy=strategy)
                data_frame[name] = np.array(imp.fit_transform(np.array(data_frame[name]).reshape(1, -1))).reshape(-1, 1)

        if encoder == 'on':
            for name in columns_encoder:
                label_encode = LabelEncoder()
                data_frame[name] = label_encode.fit_transform(data_frame[name])

        if drop_null == 'on':
            data_frame = data_frame.dropna()

        global x, y

        x = data_frame.iloc[:, :-1]
        y = data_frame.iloc[:, -1:]
        print(data_frame)
        print(data_frame.isna().sum())

    # ------------------------------------------------------------------------------------------------------------------
    # Imputer

    label11 = customtkinter.CTkLabel(pre_frame, text="Imputer", font=('Lucida Sans', 16))
    label11.grid(row=0, column=0, padx=20, pady=20)

    combo = customtkinter.CTkOptionMenu(pre_frame, values=["none", "mean", "median", "most_frequent"],
                                        command=combo_callback, )
    combo.grid(row=0, column=1, padx=20, pady=20)

    option1 = customtkinter.CTkOptionMenu(pre_frame, values=columns_name,
                                          command=option_callback).grid(row=0, column=2, padx=20, pady=20)

    # ------------------------------------------------------------------------------------------------------------------
    # encoder

    label12 = customtkinter.CTkLabel(pre_frame, text="Label encoder", font=('Lucida Sans', 16))
    label12.grid(row=1, column=0, padx=20, pady=20)

    check_var = customtkinter.StringVar(value="off")
    checkbox = customtkinter.CTkCheckBox(pre_frame, text="Label Encoder", font=('Lucida Sans', 16),
                                         command=checkbox_event,
                                         variable=check_var, onvalue="on", offvalue="off")
    checkbox.grid(row=1, column=1,
                  padx=20, pady=20)

    option = customtkinter.CTkOptionMenu(pre_frame, values=columns_name,
                                         command=option1_callback).grid(row=1, column=2, padx=20, pady=20)
    # ------------------------------------------------------------------------------------------------------------------
    # drop null

    check_var1 = customtkinter.StringVar(value="off")

    checkbox1 = customtkinter.CTkCheckBox(pre_frame, text="Drop Null", font=('Lucida Sans', 16),
                                          command=checkbox_event1,
                                          variable=check_var1, onvalue="on", offvalue="off")
    checkbox1.grid(row=2, column=0,
                   padx=20, pady=20)

    # ------------------------------------------------------------------------------------------------------------------

    submit_btn = customtkinter.CTkButton(pre_frame, text="Submit", font=('Lucida Sans', 20), command=submit1)
    submit_btn.grid(row=3, column=0, padx=20, pady=20)

    pre_window.mainloop()


pre_button = customtkinter.CTkButton(main_frame, text="preprocessing", font=('Lucida Sans', 16), command=pre)
pre_button.grid(row=2, column=0, padx=20, pady=20)

# ----------------------------------------------------------------------------------------------------------------------

button1 = customtkinter.CTkButton(main_frame, text='open', font=('Lucida Sans', 16), command=location)
button1.grid(row=0, column=1, padx=20, pady=20, )
# ----------------------------------------------------------------------------------------------------------------------

label2 = customtkinter.CTkLabel(main_frame, text='First 50 rows ', font=('Lucida Sans', 24))
label2.grid(row=1, column=0, padx=20, pady=20, )

csv_frame = customtkinter.CTkScrollableFrame(master=frame, width=550, height=200 )
csv_frame.grid(row=2, column=0, padx=15, pady=20, sticky=tk.W)
# ----------------------------------------------------------------------------------------------------------------------

describe_frame = customtkinter.CTkScrollableFrame(frame, width=550, height=150, orientation="horizontal")
describe_frame.grid(row=3, column=0, padx=20, pady=20, sticky=tk.W)


# ----------------------------------------------------------------------------------------------------------------------
# display the data

def tabel():
    data = pd.read_csv(csv_path, header=0)

    global data_frame, columns_name
    data_frame = data
    columns_name = data.columns

    with open(csv_path, "r", newline="") as passfile:
        reader = csv.reader(passfile)
        data_list = list(reader)

    for i, row in enumerate(data_list, start=0):
        if i == 50:
            break
        for col in range(len(row)):
            customtkinter.CTkLabel(csv_frame, text=row[col], font=('Lucida Sans', 16)).grid(row=i, column=col)
    z = 0
    for i in data:
        customtkinter.CTkLabel(describe_frame,
                               text=f'{i} : count : {data[i].count()} : '
                                    f'max : {data[i].max()} : '
                                    f'min : {data[i].min()}'
                                    f' : mean : {data[i].mean()}',
                               font=('Lucida Sans', 14), ).grid(row=z, sticky=tk.W)
        z = z + 1


# ----------------------------------------------------------------------------------------------------------------------

button2 = customtkinter.CTkButton(main_frame, text='display csv', font=('Lucida Sans', 16), command=tabel)
button2.grid(row=1, column=1, padx=20, pady=20, )

# ----------------------------------------------------------------------------------------------------------------------
# slide_bar for train size

slider_frame = customtkinter.CTkFrame(frame)
slider_frame.grid(row=0, column=1, padx=20, pady=20, sticky=tk.W)

label3 = customtkinter.CTkLabel(slider_frame, text=f'Train Size : ', font=('Lucida Sans', 16))
label3.grid(row=0, column=0, padx=20, pady=20, sticky=tk.W)


def slider_event(value):
    global train_size

    train_size = round(value, 2)

    label32 = customtkinter.CTkLabel(slider_frame, text=f'Train Size : {train_size}', font=('Lucida Sans', 16))
    label32.grid(row=0, column=0, padx=20, pady=20, sticky=tk.W)
    global x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size)


slider_bar = customtkinter.CTkSlider(slider_frame, width=400, from_=0, to=.99, command=slider_event)
slider_bar.grid(row=1, column=0, padx=20, pady=20, sticky=tk.W)
slider_bar.set(0)


def slider_event1(value):
    global tol

    tol = round(value, 2)

    label321 = customtkinter.CTkLabel(slider_frame, text=f'lamda : {tol}', font=('Lucida Sans', 16))
    label321.grid(row=2, column=0, padx=20, pady=20, sticky=tk.W)


label31 = customtkinter.CTkLabel(slider_frame, text=f'lamda : ', font=('Lucida Sans', 16))
label31.grid(row=2, column=0, padx=20, pady=20, sticky=tk.W)

slider_bar1 = customtkinter.CTkSlider(slider_frame, width=400, from_=0, to=.99, command=slider_event1)
slider_bar1.grid(row=3, column=0, padx=20, pady=20, sticky=tk.W)
slider_bar1.set(0)


# ----------------------------------------------------------------------------------------------------------------------

def train():
    global tol, m
    if tol > 0:
        m = Lasso()

    else:
        m.fit(x_train, y_train)

    m.fit(x_train, y_train)
    score_train = m.score(x_train, y_train)
    score_test = m.score(x_test, y_test)
    y_pred = m.predict(x_test)

    global mse, rmse, mae
    mse = round(mean_squared_error(y_test, y_pred), 2)
    rmse = round(np.sqrt(mse), 2)
    mae = round(mean_absolute_error(y_test, y_pred), 2)
    customtkinter.CTkLabel(train_frame, text=f'Train Score : {score_train}', font=('Lucida Sans', 24)).grid(row=1,
                                                                                                            column=0,
                                                                                                            padx=20,
                                                                                                            pady=20,
                                                                                                            sticky=tk.NW)
    customtkinter.CTkLabel(train_frame, text=f'Test Score : {score_test}', font=('Lucida Sans', 24)).grid(row=2,
                                                                                                          column=0,
                                                                                                          padx=20,
                                                                                                          pady=20,
                                                                                                          sticky=tk.NW)
    customtkinter.CTkLabel(train_frame, text=f"MSE : {mse} ,RMSE : {rmse},MAS : {mae}", font=('Lucida Sans', 16)).grid(
        row=3,
        column=0,
        padx=10,
        pady=10, )


def test():
    pre_window = customtkinter.CTkToplevel(frame)
    pre_window.geometry("720x500")
    pre_window.title("Test")
    test_frame = customtkinter.CTkFrame(master=pre_window)
    test_frame.pack(pady=20, padx=20, fill="both", expand=True)
    label111 = customtkinter.CTkLabel(test_frame, text="Enter Data : ", font=('Lucida Sans', 16))
    label111.grid(row=0, column=0, padx=20, pady=20)

    def submit():
        value = entry.get()
        value = value.split(',')
        for i, item in enumerate(value):
            value[i] = float(item)
        global m
        customtkinter.CTkLabel(test_frame, text=m.predict(np.array(value).reshape(1, -1)),
                               font=('Lucida Sans', 16)).grid(row=2, column=0,
                                                              padx=20, pady=20)

    entry = customtkinter.CTkEntry(test_frame, font=('Lucida Sans', 16))
    entry.grid(row=0, column=1, padx=20, pady=20)
    submit_btn = customtkinter.CTkButton(test_frame, text="submit", font=('Lucida Sans', 16), command=submit)
    submit_btn.grid(row=1, column=0, padx=20, pady=20)


train_frame = customtkinter.CTkFrame(frame, )
train_frame.grid(row=2, column=1, padx=20, pady=20, sticky=tk.NW)
customtkinter.CTkLabel(train_frame, text=f'Train Score : ', font=('Lucida Sans', 24)).grid(row=1, column=0,
                                                                                           padx=20,
                                                                                           pady=20,
                                                                                           sticky=tk.NW)
customtkinter.CTkLabel(train_frame, text=f'Test Score : ', font=('Lucida Sans', 24)).grid(row=2, column=0,
                                                                                          padx=20, pady=20,
                                                                                          sticky=tk.NW)
button3 = customtkinter.CTkButton(train_frame, text='Train', font=('Lucida Sans', 24), command=train)
button3.grid(row=0, column=0, padx=20, pady=20, sticky=tk.NW)

button4 = customtkinter.CTkButton(train_frame, text='Test', font=('Lucida Sans', 24), command=test)
button4.grid(row=0, column=1, padx=20, pady=20, )

# ----------------------------------------------------------------------------------------------------------------------
draw_frame = customtkinter.CTkFrame(frame)
draw_frame.grid(row=3, column=1, padx=20, pady=20, sticky=tk.NW)


def predict():
    plt.scatter(x, y, c='red', alpha=0.5)
    plt.plot(x_train, m.predict(x_train))
    plt.show()


def cls():
    pre_window1 = customtkinter.CTkToplevel(frame)
    pre_window1.geometry("720x500")
    pre_window1.title("Classification")
    cls_frame = customtkinter.CTkFrame(master=pre_window1)
    cls_frame.pack(pady=20, padx=20, fill="both", expand=True)

    def knn():
        global m
        m = KNeighborsClassifier(n_neighbors=n_neighbors)
        m.fit(x_train, y_train)
        score_train = m.score(x_train, y_train)
        score_test = m.score(x_test, y_test)

        customtkinter.CTkLabel(cls_frame, text=f'Train Score : {score_train}', font=('Lucida Sans', 16)).grid(row=3,
                                                                                                              column=0,
                                                                                                              padx=20,
                                                                                                              pady=20,
                                                                                                              sticky=tk.NW)
        customtkinter.CTkLabel(cls_frame, text=f'Test Score : {score_test}', font=('Lucida Sans', 16)).grid(row=3,
                                                                                                            column=1,
                                                                                                            padx=20,
                                                                                                            pady=20,
                                                                                                            sticky=tk.NW)

    def svm():
        global m
        m = LinearSVC()
        m.fit(x_train, y_train)
        score_train = m.score(x_train, y_train)
        score_test = m.score(x_test, y_test)

        customtkinter.CTkLabel(cls_frame, text=f'Train Score : {score_train}', font=('Lucida Sans', 16)).grid(row=3,
                                                                                                              column=0,
                                                                                                              padx=20,
                                                                                                              pady=20,
                                                                                                              sticky=tk.NW)
        customtkinter.CTkLabel(cls_frame, text=f"Test Score : {score_test}", font=('Lucida Sans', 16)).grid(row=3,
                                                                                                            column=1,
                                                                                                            padx=20,
                                                                                                            pady=20,
                                                                                                            sticky=tk.NW)

    button49 = customtkinter.CTkButton(cls_frame, text="KNN", font=('Lucida Sans', 16), command=knn)
    button49.grid(row=0, column=0, padx=20, pady=20, )

    button59 = customtkinter.CTkButton(cls_frame, text="SVM", font=('Lucida Sans', 16), command=svm)
    button59.grid(row=0, column=1, padx=20, pady=20, )

    def n_slide(value):
        global n_neighbors

        n_neighbors = round(value)

        label32 = customtkinter.CTkLabel(cls_frame, text=f'Num Of Neighbors : {n_neighbors}', font=('Lucida Sans', 16))
        label32.grid(row=2, column=0, padx=20, pady=20, sticky=tk.W)

    neighbors_slider = customtkinter.CTkSlider(cls_frame, width=100, from_=1, to=10, command=n_slide)
    neighbors_slider.grid(row=1, column=0, padx=20, pady=20)
    neighbors_slider.set(5)


button4 = customtkinter.CTkButton(draw_frame, text='Draw And Predict', font=('Lucida Sans', 16), command=predict)
button4.grid(row=0, column=0, padx=20, pady=20, )
button42 = customtkinter.CTkButton(draw_frame, text='Classification', font=('Lucida Sans', 16), command=cls)
button42.grid(row=0, column=1, padx=20, pady=20, )

root.mainloop()
