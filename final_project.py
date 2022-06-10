#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, dialog
from PIL import ImageTk, Image, ImageDraw
import face_recognition
from IPython.display import display
import threading


# In[2]:


known_list = [
    {
        'name': 'aoi',
        'filename': '2.jpeg',
        'encode': None,
    },
    {
        'name': 'ko',
        'filename': '25.jpeg',
        'encode': None,        
    },
]
unknown_list = ['8.jpeg', '26.jpeg', '5.jpeg']
tolerance = 0.6


# In[14]:


def load_img_button():
    """push the button and load the photo """
    global file_path, img_tk, img_name
    
    #record the file path
    file_path = filedialog.askopenfilename(parent=root, 
                                          title = 'choose photo',
                                          initialdir = 'D:\Downloads',
                                          filetypes = [("JPG files", "*.jpg"), ("JPEG files", "*.jpeg")])
    #open the image and alter the size
    img = Image.open(file_path).resize((200, 250))
    #show image on the GUI
    img_tk = ImageTk.PhotoImage(img)
    img_label = tk.Label(root, bg = 'gray94', fg = 'blue',image = img_tk)
    img_label.grid(row = 0, column = 1, rowspan = 2)
    
    #enter the name of image
    img_name = tk.Entry(frame_row1)
    img_name.grid(row = 0, column = 1)
    add_name = tk.Button(frame_row1, text = 'Enter name',command =  known_face_list)
    add_name.grid(row = 0, column = 0)
    
    #enter the unknown image to the unknown_list
    no_button = tk.Button(frame_row1, text = 'Unknown name', command = unknown_face_list)
    no_button.grid(row = 1, column = 0, columnspan = 2, sticky = tk.N+tk.S+tk.W+tk.E)
    
def show_known_list():
    """show total known face"""
    i = 0
    known_factor = len(known_list)
    for data in known_list:
        img = Image.open(data['filename']).resize((500//known_factor, 300))
        #alter the size of image
        #show image on the GUI
        im = tk.Label(root)
        im.photo = ImageTk.PhotoImage(img)
        tk.Label(frame_row4, text = data['name'], compound = 'bottom',font = ('微軟雅黑',20), image = im.photo).                pack(side = 'left')
        i = i + 1 
def show_unknown_list():
    """show total known face"""
    unknown_factor = len(unknown_list)
    for data in unknown_list:
        img = Image.open(data).resize((500//unknown_factor, 300))
        #alter the size of image
        #show image on the GUI
        im = tk.Label(root)
        im.photo = ImageTk.PhotoImage(img)
        tk.Label(frame_row4, image = im.photo).pack(side = 'left')
        
def known_face_list():
    """Add known image in the known_list"""
    dic_name = {'name': img_name.get(),
                'filename': file_path,
                'encode': None}
    known_list.append(dic_name)
    tk.messagebox.showinfo("Successfully", "Successfully Add in the known list")
    
def unknown_face_list():
    """Add unknown image in the unknown_list"""
    unknown_list.append(file_path)
    tk.messagebox.showinfo("Successfully", "Successfully Add in the unknown list")
    
RED_COLOR = (200, 58, 76)
WHITE_COLOR = (255, 255, 255)

def draw_locations(img, match_results):
    """draw the rectangle on the face"""
    for match_result in match_results:
        y1, x2, y2, x1 = match_result['location']
        cv2.rectangle(img, (x1, y1), (x2, y2), RED_COLOR, 2)
        cv2.rectangle(img, (x1, y2 + 35), (x2, y2), RED_COLOR, cv2.FILLED)
        cv2.putText(img, match_result['name'], (x1 + 10, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, WHITE_COLOR, 2)

def recognize_face():
    """recognize whose face"""
    for data in known_list:
        img = cv2.imread(data['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data['encode'] = face_recognition.face_encodings(img)[0]
    known_face_encodes = [data['encode'] for data in known_list]
    i = 0
    unknown_factor = len(unknown_list)
    for fn in unknown_list:
        match_results = []

        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cur_face_locs = face_recognition.face_locations(img)
        cur_face_encodes = face_recognition.face_encodings(img, cur_face_locs)

        for cur_face_encode, cur_face_loc in zip(cur_face_encodes, cur_face_locs):
            face_distance_list = face_recognition.face_distance(known_face_encodes, cur_face_encode)

            min_distance_index = np.argmin(face_distance_list)
            if face_distance_list[min_distance_index] < tolerance:
                name = known_list[min_distance_index]['name']
            else:
                name = 'unknown'

            match_results.append({
                'name': name,
                'location': cur_face_loc,
            })

            draw_locations(img, match_results)
            im = tk.Label(root)
            im.photo = ImageTk.PhotoImage(Image.fromarray(img).resize((500//unknown_factor, 300)))
            tk.Label(frame_row4, text = known_list[min_distance_index]['name'], compound = 'bottom',image = im.photo)                    .pack(side = 'left')
            i = i + 1   
def remove():
    """remove the widget in the frame"""
    for widget in frame_row4.winfo_children():
        widget.destroy()


# In[16]:


root = tk.Tk()
root.title('img')
root.geometry('500x700')

#design frame
frame_row0 = tk.Frame(root, width = 300, height = 40)
frame_row0.grid(row = 0, column = 0)
frame_row0.propagate(0)

frame_row1 = tk.Frame(root, width = 300, height = 210)
frame_row1.grid(row = 1, column = 0)
frame_row1.propagate(0)

frame_right = tk.Frame(root, width = 200, height = 250)
frame_right.grid(row = 0, column = 1, rowspan = 2)
frame_right.propagate(0)

frame_row2 = tk.Frame(root, width = 500, height = 40)
frame_row2.grid(row = 2,column = 0, columnspan = 2)

frame_row3 = tk.Frame(root, width = 500, height = 40)
frame_row3.grid(row = 3, column = 0, columnspan = 2)

frame_row4 = tk.Frame(root, width = 500, height = 300)
frame_row4.grid(row = 4, column = 0, columnspan = 2)

#function 
add_data = tk.Button(root, text = 'Add image', command = load_img_button)
add_data.grid(row = 0, column = 0,  sticky = tk.W+tk.E)

known_button = tk.Button(root, text = 'Show known image', command = show_known_list)
known_button.grid(row = 2, column = 0, sticky = tk.N+tk.S+tk.W+tk.E)

unknown_button = tk.Button(root, text = 'Show unknown image', command = show_unknown_list)
unknown_button.grid(row = 2, column = 1, sticky = tk.N+tk.S+tk.W+tk.E)

compare = tk.Button(root, text = 'Comparation', command = recognize_face)
compare.grid(row = 3, column = 0, columnspan = 2, sticky = tk.N+tk.S+tk.W+tk.E)

remove = tk.Button(root, text = 'Please remove when show the new results', command = remove)
remove.grid(row = 5, column = 0, columnspan = 2, sticky = tk.N+tk.S+tk.W+tk.E)

root.mainloop()


# In[ ]:





# In[ ]:




