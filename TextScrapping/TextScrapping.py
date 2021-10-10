import easyocr
from PIL import Image,ImageDraw



def draw_boxes(image,bounds,color='yellow',width=2):
    draw=ImageDraw.Draw(image)
    for i in bounds:
        p1,p2,p3,p4=i[0]
        draw.line([*p1, *p2, *p3, *p4, *p1],fill=color,width=width)
    return image.show()	


reader=easyocr.Reader(['en'],gpu=False)
img=Image.open('test.jpg')

bounds=reader.readtext(img)

draw_boxes(img,bounds)

for i in bounds:
    print(i[1])
