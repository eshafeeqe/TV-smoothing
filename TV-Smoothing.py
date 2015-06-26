import numpy as np
from matplotlib import pyplot as plt
import cv2
import sys, getopt


def PimalDualSmoothMain(image):

    sigma = 0.5
    tow = 0.25
    alpha = 100000
    error_limit = 0.0025
    lamda = 1
    img = np.true_divide(np.double(image),255)
    p0,p1 = (np.zeros(img.shape) for i in range(2))
    evolve_mask = np.ones(img.shape)
    u = img
    error = 10
    
    while error > error_limit:
        LyU = np.roll(u,-1,axis=0) #Left translation w.r.t. the y-direction
        LxU = np.roll(u,-1,axis=1) #Left translation w.r.t. the x-direction
        
        grad_Ux = (LxU - u )
        grad_Uy = (LyU - u )
        
        p0 = p0 + (sigma*grad_Ux)
        p1 = p1 + (sigma*grad_Uy)
        
        p_abs = (p0**2 + p1**2)**(0.5)
        
        th = ((lamda*sigma*(sigma+2*alpha))/alpha)**(0.5)        
        zero_cordinates = p_abs>th        
        evolve_mask[zero_cordinates] = 0 
        
        p0 = p0*evolve_mask
        p1 = p1*evolve_mask
        
        p0 = (p0*2*alpha)/(sigma + 2*alpha)
        p1 = (p1*2*alpha)/(sigma + 2*alpha)
        
        RxPx = np.roll(p0,1,axis=1)
        RyPy = np.roll(p1,1,axis=0)
        
        div_p = (p0-RxPx) + (p1-RyPy)
        
        prev_u = u
        
        u = u + tow*(div_p)
        
        u = (u + 2*tow*img)/(1+2*tow)
        
        theta = float(1)/((1+4*tow)**(0.5))
        
        tow = theta*tow
        sigma = sigma/theta
        
        error = (((np.abs(prev_u - u)).flatten()).sum())/(u.shape[0]*u.shape[1])
        u = u + (theta*(u-prev_u))
        
    smoothed = (u*255).astype(np.uint8)
    
    return smoothed
    
if __name__ == '__main__':
    
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:o:",["ifile="])
    except getopt.GetoptError:
      print 'TV-Smoothing.py <inputfile>'
      sys.exit(2) 
      
    if len(args)==0 or len(args)>1:
      print 'usage: TV-Smoothing.py <inputfile>'
      sys.exit() 
       
    inputfile = args[0]   
    img = cv2.imread(inputfile)
    smoothed = PimalDualSmoothMain(img)
    cv2.imwrite("smoothed.png",smoothed)
