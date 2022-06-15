#!/usr/bin/env python
# coding: utf-8

# In[111]:


import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, dialog
from PIL import ImageTk, Image, ImageDraw
import face_recognition
from IPython.display import display
import threading


# In[112]:


known_list = [
    {
        'name': 'Aragaki_Yui',
        'filename': '.\image\Aragaki_Yui.jpeg',
        'encode': None,
    },
    {
        'name': 'Walter_White',
        'filename': '.\image\Walter_White.jpg',
        'encode': None,        
    },
]
unknown_list = ['.\image\Walter_White_test.jpg','.\image\Aragaki_Yui_test.jpg', '.\image\IU_test.jpg']
threshold = 0.6


# In[113]:


def load_img_button():
    """push the button and load the photo """
    global file_path, img_tk, img_name
    
    #record the file path
    file_path = filedialog.askopenfilename(parent=root, 
                                          title = 'choose photo',
                                          initialdir = 'D:',
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
    known_factor = len(known_list)
    for data in known_list:
        #alter the size of image
        img = Image.open(data['filename']).resize((500//known_factor, 300))
        
        #show image on the GUI
        im = tk.Label(root)
        im.photo = ImageTk.PhotoImage(img)
        tk.Label(frame_row4, text = data['name'], compound = 'bottom',font = ('微軟雅黑',20), image = im.photo).                pack(side = 'left')

def show_unknown_list():
    """show total known face"""
    unknown_factor = len(unknown_list)
    for data in unknown_list:
        #alter the size of image
        img = Image.open(data).resize((500//unknown_factor, 300))
        
        #show image on the GUI
        im = tk.Label(root)
        im.photo = ImageTk.PhotoImage(img)
        tk.Label(frame_row4, image = im.photo).pack(side = 'left')
        
def known_face_list():
    """Add known image in the known_list"""
    #add image information in the known list
    dic_name = {'name': img_name.get(),
                'filename': file_path,
                'encode': None}
    known_list.append(dic_name)
    #show the successful messages of adding image in the known list
    tk.messagebox.showinfo("Successfully", "Successfully Add in the known list")
    
def unknown_face_list():
    """Add unknown image in the unknown_list"""
    #add image's path in the unknown lsit
    unknown_list.append(file_path)
    #show the successful messages of adding image in the known list
    tk.messagebox.showinfo("Successfully", "Successfully Add in the unknown list")
    


def draw_locations(img, match_results):
    """draw the rectangle on the face"""
    
    for match_result in match_results:
        y1, x2, y2, x1 = match_result['location']
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(img, (x1, y2 + 35), (x2, y2), (255, 0, 0), cv2.FILLED)
        cv2.putText(img, match_result['name'], (x1 + 10, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)

def thread():
    """use thread to process multiple work"""
    threading.Thread(target = recognize_face).start()
    
def recognize_face():
    """recognize whose face"""
    #turn image in the RGB code
    for data in known_list:
        img = cv2.imread(data['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data['encode'] = face_recognition.face_encodings(img)[0]
    known_face_encodes = [data['encode'] for data in known_list]
    
    #the width factor of showing image 
    unknown_factor = len(unknown_list)
    
    #compare the unknown_list and known_list 
    for photo in unknown_list:
        match_results = []
        
        #read the photo in the unkonwn_list and convert its code
        img = cv2.imread(photo)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        #record location and codes of the face in the unknown image 
        unknown_face_locs = face_recognition.face_locations(img)
        unknown_face_encodes = face_recognition.face_encodings(img, unknown_face_locs)

        for cur_face_encode, cur_face_loc in zip(unknown_face_encodes, unknown_face_locs):
            #compute the Euclidean Distance between the known_face and unkonwn_face
            face_distance_list = face_recognition.face_distance(known_face_encodes, cur_face_encode)
            
            #the smaller distance it is, the more smiliarity they are
            #set an threshold to evaluate the similiarity
            min_distance_index = np.argmin(face_distance_list)
            if face_distance_list[min_distance_index] < threshold:
                name = known_list[min_distance_index]['name']
            else:
                name = 'unknown'

            match_results.append({
                'name': name,
                'location': cur_face_loc,
                'distance': face_distance_list[min_distance_index]
            })
            #draw the rectangle on the face
            draw_locations(img, match_results)
            #show the results on the GUI
            im = tk.Label(root)
            im.photo = ImageTk.PhotoImage(Image.fromarray(img).resize((500//unknown_factor, 300)))
            tk.Label(frame_row4,
                     text = str([name, round(face_distance_list[min_distance_index], 2)]), 
                     compound = 'bottom',image = im.photo).pack(side = 'left') 
def remove():
    """remove the widget in the frame"""
    for widget in frame_row4.winfo_children():
        widget.destroy()


# In[114]:


root = tk.Tk()
root.title('face_recognition')
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

#GUI button
add_data = tk.Button(root, text = 'Add image', command = load_img_button)
add_data.grid(row = 0, column = 0,  sticky = tk.W+tk.E)

known_button = tk.Button(root, text = 'Show known image', command = show_known_list)
known_button.grid(row = 2, column = 0, sticky = tk.N+tk.S+tk.W+tk.E)

unknown_button = tk.Button(root, text = 'Show unknown image', command = show_unknown_list)
unknown_button.grid(row = 2, column = 1, sticky = tk.N+tk.S+tk.W+tk.E)

compare = tk.Button(root, text = 'Comparation', command = thread)
compare.grid(row = 3, column = 0, columnspan = 2, sticky = tk.N+tk.S+tk.W+tk.E)

remove = tk.Button(root, text = 'Please remove old results before show new results', command = remove)
remove.grid(row = 5, column = 0, columnspan = 2, sticky = tk.N+tk.S+tk.W+tk.E)

root.mainloop()


# In[ ]:





# In[ ]:




