from tkinter import *
import numpy
import csv
import math
import operator
import random
import pickle

'''
#2 == 29
model.predict([[90,	79,	72,	0,	0,	0,	0,	0,	0,	-1.4,	-0.1,	-29,	0,	2.2,	-1.7,	-1.4]])
#10 == 3
model.predict([[138,	75,	40,	76,	100,	28,	60,	0,	-2.8,	2.5,	-2.5,	-28.5,	6.5,	-2.4,	2.2,	3.4]])
#11 == 6
model.predict([[100,	84,	36,	0,	24,	60,	0,	24,	-6.4,	1.7,	-2.9,	-39,	0,	3.4,	5.9,	2.2]])  
'''

def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((x, dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

def getResponse(neighbors,Y):
	classVotes = {}
	for x in range(len(neighbors)):
		response =numpy.asscalar(Y[neighbors[x]]);
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]


def loadDataset(split,X,Y, X_train=[] , Y_train=[],  X_test=[],  Y_test=[]):
    c=0
    for i in range(0,X.shape[0]):
        if c/X.shape[0] < 1-split and i%4==2 and Y[i]!=7 and Y[i]!=8 and Y[i]!=11 and Y[i]!=12:
            if c<15 and Y[i]==1:
                X_test.append(X[i])
                Y_test.append(Y[i])
                c=c+1
            elif c>=15:
                X_test.append(X[i])
                Y_test.append(Y[i])
                c=c+1
            else:
                X_train.append(X[i])
                Y_train.append(Y[i])
        else:
            X_train.append(X[i])
            Y_train.append(Y[i])
    return c

#create reduced feature matrix
reader=csv.reader(open("reduced_features.csv","r"),delimiter=",")
X=list(reader)
X=numpy.array(X)
X=X.astype(numpy.float)


#create result vector
reader=csv.reader(open("target_output.csv","r"),delimiter=",")
Y=list(reader)
Y=numpy.array(Y)
Y=Y.astype(numpy.int)

no_of_columns=X.shape[1]

X_train=[]
Y_train=[]
X_test=[]
Y_test=[]

c=loadDataset(0.8,X,Y, X_train , Y_train,  X_test,  Y_test)


def calculation(txt1,txt2,txt3,txt4,txt5,txt6,txt7,txt8,txt9,txt10,txt11,txt12,txt13,txt14,txt15,txt16):
    k=4
    newX=[float(txt1), float(txt2), float(txt3), float(txt4), float(txt5), float(txt6), float(txt7), float(txt8), float(txt9), float(txt10), float(txt11), float(txt12), float(txt13), float(txt14), float(txt15), float(txt16)]
    neighbors = getNeighbors(X,newX,k)
    result = getResponse(neighbors,Y)
    txtoutput['text'] = getanswer(int(result))

def getanswer(answer):
    switcher = {
        1: "Normal",
        2: "Ischemic Changes(Coronary Artery Disease)",
        3: "Old Anterior Myocardial Infarction",
        4: "Old Inferior Myocardial Infarction",
        5: "Synus tachycardy",
        6: "Sinus bradycardy",
        7: "Ventricular Premature Contraction(PVC)",
        8: "Supraventricular Premature Contraction",
        9: "Left bundle branch block",
        10: "Right bundle branch block",
        11: "Left ventricule hypertropy",
        12: "Atrial Fibrillation or Flutter",
        13: "Others"
    }
    return switcher.get(answer, "Invalid Input")


root = Tk()
root.title("ECG Test")
canvas = Canvas(root, height=600, width=800)
canvas.pack()

heading = Label(root, text="ECG Classification", font=('Algerian',25))
heading.place(relx=0.3,relheight=0.1,relwidth=0.4)
inframe = Frame(root, bg='#80c1ff')
inframe.place(relx=0.05, rely=0.08, relwidth=0.9, relheight=0.64)
lbl1 = Label(inframe, text="QRS Duration",bg='#80c1ff')
txt1 = Entry(inframe)
lbl2 = Label(inframe, text="J value",bg='#80c1ff')
txt2 = Entry(inframe)
lbl3 = Label(inframe, text="Q wave avg width chDI",bg='#80c1ff')
txt3 = Entry(inframe)
lbl4 = Label(inframe, text="S wave avg width chV1",bg='#80c1ff')
txt4 = Entry(inframe)
lbl5 = Label(inframe, text="S' wave avg width chV1",bg='#80c1ff')
txt5 = Entry(inframe)
lbl6 = Label(inframe, text="Q wave avg width chV2",bg='#80c1ff')
txt6 = Entry(inframe)
lbl7 = Label(inframe, text="R wave avg width chV2",bg='#80c1ff')
txt7 = Entry(inframe)
lbl8 = Label(inframe, text="Diphasic derv of T wave chV2",bg='#80c1ff')
txt8 = Entry(inframe)
lbl9 = Label(inframe, text="Amplitude of R wave chDI",bg='#80c1ff')
txt9 = Entry(inframe)
lbl10 = Label(inframe, text="Amplitude of P wave chDI",bg='#80c1ff')
txt10 = Entry(inframe)
lbl11 = Label(inframe, text="Amplitude of P wave chAVR",bg='#80c1ff')
txt11 = Entry(inframe)
lbl12 = Label(inframe, text="QRSA value of chAVR",bg='#80c1ff')
txt12 = Entry(inframe)
lbl13 = Label(inframe, text="Amplitude of S wave chV1",bg='#80c1ff')
txt13 = Entry(inframe)
lbl14 = Label(inframe, text="Amplitude of P wave chV2",bg='#80c1ff')
txt14 = Entry(inframe)
lbl15 = Label(inframe, text="Amplitude of P wave chV4",bg='#80c1ff')
txt15 = Entry(inframe)
lbl16 = Label(inframe, text="Amplitude of P wave chV6",bg='#80c1ff')
txt16 = Entry(inframe)
predict = Button(inframe, text="Classify", font=('Arial Black',14), command=lambda: calculation(txt1.get(),txt2.get(),txt3.get(),txt4.get(),txt5.get(),txt6.get(),txt7.get(),txt8.get(),txt9.get(),txt10.get(),txt11.get(),txt12.get(),txt13.get(),txt14.get(),txt15.get(),txt16.get()))

#photo = PhotoImage(file="barca.gif")
#canvas.create_image(0, 0, anchor=S, image=photo)

#phlabel = Label(root,image=photo)
#phlabel.pack()
#phlabel.grid(row=2)
lbl1.grid(row=0,sticky=E,pady=10,padx=10)
txt1.grid(row=0,column=1,pady=10,padx=10)
lbl2.grid(row=1,sticky=E,pady=10,padx=10)
txt2.grid(row=1,column=1,padx=10,pady=10)
lbl3.grid(row=2,sticky=E,pady=10,padx=10)
txt3.grid(row=2,column=1,pady=10,padx=10)
lbl4.grid(row=3,sticky=E,pady=10,padx=10)
txt4.grid(row=3,column=1,padx=10,pady=10)
lbl5.grid(row=4,sticky=E,pady=10,padx=10)
txt5.grid(row=4,column=1,pady=10,padx=10)
lbl6.grid(row=5,sticky=E,pady=10,padx=10)
txt6.grid(row=5,column=1,padx=10,pady=10)
lbl7.grid(row=6,sticky=E,pady=10,padx=10)
txt7.grid(row=6,column=1,pady=10,padx=10)
lbl8.grid(row=7,sticky=E,pady=10,padx=10)
txt8.grid(row=7,column=1,padx=10,pady=10)

lbl9.grid(row=0,column=4,sticky=E)
txt9.grid(row=0,column=5,padx=10,pady=10)
lbl10.grid(row=1,column=4,sticky=E)
txt10.grid(row=1,column=5,padx=10,pady=10)
lbl11.grid(row=2,column=4,sticky=E)
txt11.grid(row=2,column=5,pady=10,padx=10)
lbl12.grid(row=3,column=4,sticky=E)
txt12.grid(row=3,column=5,padx=10,pady=10)
lbl13.grid(row=4,column=4,sticky=E)
txt13.grid(row=4,column=5,pady=10,padx=10)
lbl14.grid(row=5,column=4,sticky=E)
txt14.grid(row=5,column=5,padx=10,pady=10)
lbl15.grid(row=6,column=4,sticky=E)
txt15.grid(row=6,column=5,pady=10,padx=10)
lbl16.grid(row=7,column=4,sticky=E)
txt16.grid(row=7,column=5,padx=10,pady=10)
predict.grid(row=8,column=2,pady=10)

outframe = Frame(root, bg='#80c1ff')
outframe.place(relx=0.05, rely=0.75, relwidth=0.9, relheight=0.2)

lbloutput= Label(outframe, text="OUTPUT",font=("Courier", 16),bg='#1F1142',fg='#B4A1E3')
lbloutput.place(relheight=0.2,relwidth=0.4,relx=0.3)
txtoutput = Label(outframe,font=("Courier", 16))
txtoutput.place(relx=0.05, rely=0.25, relheight=0.65, relwidth=0.9)

root.mainloop()# -*- coding: utf-8 -*-

