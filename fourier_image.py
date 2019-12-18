import numpy as np
import cv2
from matplotlib import pyplot as plt
import random as rd
## importation

def importation(lien):
    ''' Importe une image à partir de l'url'''
    img = cv2.imread(lien,0)
    img_float32 = np.float32(img)
    return img_float32
    
## transformée de Fourier

def fourier(img_float32):
    '''Renvoie la transformée de Fourier d'une image'''
    dft = cv2.dft(img_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return dft_shift
    
def inv_fourier(fshift):
    ''' Renvoie la transformée de Fourier inverse'''
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
    return img_back

## fonctions outils
    
def module(dft_shift):
    ''' Renvoie le module d'une transformée de Fourier'''
    dft_mod = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    return dft_mod
    
def phase(dft_shift):
    ''' Renvoie la phase d'une transformée de Fourier'''
    dft_mod = cv2.phase(dft_shift[:,:,0],dft_shift[:,:,1])
    return dft_mod
    
def seuil(img,n):
    ''' Applique un seuil à une image'''
    img_out = cv2.threshold(img,n,255,cv2.THRESH_BINARY)
    return img_out[1]

def dim(img):
    ''' Renvoie les dimensions d'une image et les coordonnées du milieu'''
    rows, cols = img.shape
    crow, ccol = rows/2 , cols/2 
    return crow, ccol, rows, cols


def sinus(dim,n,sens):
    '''Génère une image sinus avec n périodes, horizontal 0 ou vertical 1 '''
    crow, ccol, rows, cols = dim
    resu = np.zeros((rows, cols))
    if sens ==0:
        X = 0*resu[0]
        for i in range(rows):
           X[i]  = np.abs(np.sin(np.pi*(i/(rows)*n)))
        for j in range(cols):
            resu[j] = X
    if sens ==1:
        for i in range(rows):
            resu[i]  = cols * [np.abs(np.sin(np.pi*(i/(rows)*n)))]
    return resu

def bruit(img,n):
    '''Bruite une image de 100/n %'''
    crow, ccol, rows, cols = dim(img)
    resu = 0*img
    for i in range(rows):
        for j in range(cols):
            if rd.randint(1,n) == 1:
                resu[i][j] = 255
            else:
                resu[i][j] = img[i][j]
    return resu
    
def melangeur(img):
    ''' Decoupe une image en 4 parties et les mélange'''
    crow, ccol, rows, cols = dim(img)
    ccol = int(ccol)
    crow = int(crow)
    resu = 0*img
    [a,b,c,d]=[img[:crow,:ccol],img[crow:,:ccol],img[crow:,ccol:],img[:crow,ccol:]]
    [resu[:crow,:ccol],resu[crow:,:ccol],resu[crow:,ccol:],resu[:crow,ccol:]] = [b,c,d,a]
    return resu
    


## filtres

def passe_bas_carre(dim,taille):
    ''' Gènère un passe bas 2D en carré sans dégradé'''
    crow, ccol, rows, cols = dim
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[int(crow)-taille:int(crow)+taille, int(ccol)-taille:int(ccol)+taille] = 1
    return mask
    

def passe_bas_rond(dim,taille,bordure):
    ''' Génère un passe bas 2D en rond avec un dégradé'''
    crow, ccol, rows, cols = dim
    mask = np.zeros((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= (taille +bordure):
                mask[i][j] = (-1*np.sqrt((i-crow)**2 + (j-ccol)**2)+taille + bordure)/bordure
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= taille:
                mask[i][j] = 1
    return mask

def passe_haut_rond(dim,taille,bordure):
    ''' Génère un passe haut 2D en rond avec un dégradé'''
    crow, ccol, rows, cols = dim
    mask = np.ones((rows, cols, 2))
    for i in range(rows):
        for j in range(cols):
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= (taille +bordure):
                mask[i][j] = (np.sqrt((i-crow)**2 + (j-ccol)**2)-taille)/bordure
            if np.sqrt((i-crow)**2 + (j-ccol)**2) <= taille:
                mask[i][j] = 0
    return mask
    

## fonctions pour applications

def watermarkingFFT(fft,water):
    ''' Applique un tatouage à une FFT'''
    result = 0* fft
    for i in range(len(fft)):
        for j in range(len(fft[0])):
            result[i][j][0] = water[i][j]*fft[i][j][0]
            result[i][j][1] = water[i][j]*fft[i][j][1]
    return result
    
def watermarking(img,water):
    return inv_fourier(watermarkingFFT(fourier(img),water))


def filtre_bas(img,n,p):
    ''' Applique un filtre passe bas à une image'''
    return inv_fourier(fourier(img)*passe_bas_rond(dim(img),n,p))

def filtre_haut(img,n,p):
    ''' Applique un filtre passe haut à une image'''
    return inv_fourier(fourier(img)*passe_haut_rond(dim(img),n,p))


def fusion(img1,img2):
    ''' Génère une image à partir d'un module d'une image et d'une phase d'une autre image'''
    fft1 = fourier(img1)
    fft2 = fourier(img2)
    crow, ccol, rows, cols = dim(img1)
    resu = 0*fft1
    for i in range(rows):
        for j in range(cols):
            resu[i][j] = [fft1[i][j][0],fft2[i][j][1]]
    return inv_fourier(resu)
    
## affichage

def affichage(tab):
    '''Affiche une ou plusieurs images avec des titres'''
    dim=[[1,1],[1,2],[1,3],[2,2],[2,3],[2,3],[3,3],[3,3],[3,3]]
    n,p = dim[len(tab)-1]
    for i in range(len(tab)):
        number = n*100+p*10+i+1
        plt.subplot(number),plt.imshow(tab[i][0], cmap = 'gray')
        plt.title(tab[i][1], fontsize=20), plt.xticks([]), plt.yticks([])
    plt.show()
    
    
    



## code principal

img = importation('D:/Bureau/tse.jpg')


#exemple d'une image débruitée par filtre un passe bas
affichage([[img,'Image originale'],[bruit(img,4),'Image bruité'],[filtre_bas(bruit(img,4),120,40),'Image filtrée']])





















