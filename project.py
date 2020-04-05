import zipfile
from zipfile import ZipFile
from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np
from tqdm.notebook import tqdm
import PIL
from PIL import ImageDraw
from PIL import ImageFont
import string

# loading the face detection classifier
face_cascade = cv.CascadeClassifier('readonly/haarcascade_frontalface_default.xml')

def openImage(name,z):
    img = Image.open(z.open(name))
    open_cv_version=img.convert("L")
    open_cv_version.save("test.png")
    cv_img = cv.imread("test.png")
    cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
    return img, open_cv_version, cv_img

def draw_text(text,file_size):
    y = Image.new("RGB", (file_size, 50), color = (255,255,255))
    box = ImageDraw.Draw(y)
    fonttype = ImageFont.truetype("readonly/fanwood-webfont.ttf", 20)
    box.text((0,20), text, font = fonttype, color='black', fill=(0,0,0,0))
    return y

def getContactSheet(cant,shapes,file_size):
    if len(shapes)>0:
        first_image=shapes[0]
        contact_sheet=PIL.Image.new(first_image.mode, (first_image.width*cant,(first_image.height)*(((len(shapes)-1)//cant)+1)))
        x=0
        y=0
        for img in shapes:
            contact_sheet.paste(img.resize((first_image.width,first_image.height)), (x, y) )
            if x+first_image.width == contact_sheet.width:
                x=0
                y=y+first_image.height
            else:
                x=x+first_image.width
        rec_factor = file_size/contact_sheet.width
        contact_sheet = contact_sheet.resize((int(contact_sheet.width*rec_factor),int(contact_sheet.height*rec_factor)))
        return contact_sheet,'shapes'
    else:
        contact_sheet = draw_text("But were not faces in that file!",file_size)
        return contact_sheet, 'box'

def search_engine(word,data,file_size):
    images = []
    for k,v in data.items():
        if word.lower() in v['words']:
            images.append(v['base_text'])
            images.append(v['faces_sheet'])
    t_width = sum(map(lambda x:x.width,images))
    t_height = sum(map(lambda x:x.height,images))
    final_sheet=PIL.Image.new(images[0].mode, (file_size,t_height), color = (255,255,255))
    actual_height = 0
    for image in images:
        final_sheet.paste(image, (0,actual_height) )
        actual_height +=  image.height
    return final_sheet


def preprocesing(img_file,file_size):
    z = ZipFile('readonly/{}'.format(img_file))
    z_files = [zf.filename for zf in z.infolist()]
    data = {}
    for file in tqdm(z_files):
        img = openImage(file,z)
        faces_rects = face_cascade.detectMultiScale(img[2],1.35)
        shapes = [img[0].crop((rec[0],rec[1],rec[0]+rec[2],rec[1]+rec[3])) for rec in faces_rects]
        f_sheet = getContactSheet(5,shapes,file_size)
        data[file] = {
            'words':[word.lower().translate(str.maketrans('', '', string.punctuation)) for word in pytesseract.image_to_string(img[0]).split()],
            'faces_sheet': f_sheet[0],
            'faces_type':f_sheet[1],
            'base_text': draw_text("Results found in file {}".format(file),file_size)
        }
    return data

file_size= 500
data = preprocesing('small_img.zip',file_size)
img = search_engine('Christopher',data,file_size)
display(img)
data2 = preprocesing('images.zip',file_size)
img2 = search_engine('Mark',data2,file_size)
display(img2)
