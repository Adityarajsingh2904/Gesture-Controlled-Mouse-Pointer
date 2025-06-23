from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,89), Image.ANTIALIAS)
    img.save(imageName)

for i in range(0, 100):
    resizeImage("C:/Users/ayroy/Downloads/Gesture Controlled Mouse Pointer/Dataset/YoTest/yo_" + str(i) + '.png')