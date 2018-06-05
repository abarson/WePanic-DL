import sys
from PIL import Image

if __name__ == "__main__":
    
    try:
        img_pth = sys.argv[1]
        
        im = Image.open(img_pth)
        width,height = im.size

        print("%s has height %d, width %d" % (img_pth,height,width))
    except IndexError:
        sys.exit("[Usage] %s <img>" % sys.argv[0])

    except OSError:
        sys.exit("[!!] %s is a bad file" % img_pth)
