# pyoof - implementation of Optimally Oriented Flux in Python with GPU support 

import numpy as np
import cupy as cp 
import gc 
from scipy.special import jv as besselj
from tqdm import tqdm
import multiprocessing as mp 
from functools import partial 
import zarr 
from ..utils import get_chunk_coords
import time 

from numcodecs import Blosc, Zstd 

EPSILON = 1e-12

class OOF:
    def __init__(self, image, radii, **options):
        self.image = image 
        self.radii = radii # array or list of radii to evaluate OOF at 
        
        # Default options 
        if 'response_type' in options:
            self.response_type = options['response_type']
        else:
            self.response_type = 0 # what response type. 0 is max eigenvalue 
        if 'use_absolute' in options: 
            self.use_absolute = options['use_absolute']
        else:
            self.use_absolute = True # whether or not to use absolute magnitude when ordering 
        if 'normalization_type' in options:
            self.normalization_type = options['normalization_type'] # options are spherical (0), curvilinear (1), and planar (2) normalization
        else:
            self.normalization_type = 1
        if 'spacing' in options:
            self.spacing = options['spacing']
        else:
            self.spacing = (1,1,1)
        self.sigma = min(self.spacing)
        if 'calc_eigenvectors' in options:
            self.calc_eigenvectors = options['calc_eigenvectors']
        else:
            self.calc_eigenvectors = False 
        if 'do_oofofa' in options:
            self.do_oofofa = options['do_oofofa'] # whether or not to use active contour segmentation
        else:
            self.do_oofofa = False 
        self._check_radii()
        
    def _check_radii(self):
        if min(self.radii) < self.sigma and (self.normalization_type > 0):
            self.normalization_type = 0
            print('Sigma must be >= min. range to enable advanced normalization. \
            The current setting falls back to normalization_type = 0 because of \
            the undersized sigma.')
            
    def compute_oof(self):
        pad_image = True # default True 
        in_place_FFT = False # default False 
        
        xp = cp.get_array_module(self.image) # see if we do CPU or GPU 
        
        output = xp.zeros(self.image.shape)
        
        # Pad the image by reflection to get around the wrap-around artifacts 
        if pad_image:
            margin_widths = [int(xp.ceil((max(self.radii)+self.sigma*3)/self.spacing[i])) for i in [0,1,2]]
            self.image = xp.pad(self.image, ((margin_widths[0],)*2,(margin_widths[1],)*2,(margin_widths[2],)*2), 'reflect')
        else:
            margin_widths = [0, 0, 0] 
        self.image = self.image.astype('double')
        
        # Perform Fast Fourier Transform 
        imgfft = xp.fft.fftn(self.image) 
        
        # Obtain the Fourier coordinate 
        coords = ifft_shifted_coor_matrix(xp.asarray(self.image.shape), xp=xp)
        for i in range(len(coords)):
            # Square them in anticipation of the next step 
            coords[i] = coords[i]/self.image.shape[i]/self.spacing[i]
        radius = xp.sqrt(xp.square(coords[0])+xp.square(coords[1])+xp.square(coords[2])) + EPSILON
        
        # Keep track of max response eigenvector 
        # vec1_max = xp.zeros(output.shape,dtype='double')
        vec2_max = xp.zeros(output.shape,dtype='double')
        vec3_max = xp.zeros(output.shape,dtype='double') 
        
        
        for r in self.radii:
            r = float(r) 
            #print('Computing radius r = %d...'%r)
            
            # # Perform Fast Fourier Transform 
            # imgfft = xp.fft.fftn(self.image) 
            
            # Formula from:
            # "Dilated Divergence Scale-Space for Curve Analysis" 
            # Max Law et al, 2010
            ## OLD - original by Max Law - uncomment and replace in the code below to return 
            # limit as eps --> 0 of besselj(1.5,2*pi*r*eps)/eps**1.5 --> 4/3*pi*r^(3/2), why do we even have this here? 
            
            # normalization = 4*r**-1.5 * \ # This is the actual formula 
            
            # normalization = 4*xp.pi*r/(besselj(1.5,2*xp.pi*r*EPSILON)/EPSILON**1.5) * \
            normalization = 4*r**-0.5 *\
                            (r/xp.sqrt(2*r*self.sigma - self.sigma**2))**self.normalization_type
            besseljBuffer = normalization * xp.exp(-2*xp.square(xp.pi*radius*self.sigma)) / radius**1.5
            besseljBuffer = (xp.sin(2*xp.pi*r*radius)/(2*xp.pi*r*radius) - xp.cos(2*xp.pi*r*radius)) * \
                            besseljBuffer*xp.sqrt(1/(xp.pi**2*r*radius))
            besseljBuffer = xp.multiply(besseljBuffer,imgfft) 
            
            
            ## IF have enough memory, then do out-of-place FFT 
            if not in_place_FFT:
                outputfeature_11 = freq_op(xp.real(xp.fft.ifftn(coords[0]**2*besseljBuffer)), margin_widths)
                outputfeature_12 = freq_op(xp.real(xp.fft.ifftn(coords[0]*coords[1]*besseljBuffer)), margin_widths)
                outputfeature_13 = freq_op(xp.real(xp.fft.ifftn(coords[0]*coords[2]*besseljBuffer)), margin_widths)
                outputfeature_22 = freq_op(xp.real(xp.fft.ifftn(coords[1]**2*besseljBuffer)), margin_widths)
                outputfeature_23 = freq_op(xp.real(xp.fft.ifftn(coords[1]*coords[2]*besseljBuffer)), margin_widths)
                outputfeature_33 = freq_op(xp.real(xp.fft.ifftn(coords[2]**2*besseljBuffer)), margin_widths)
            else:
                ## TODO: implement in-place FFT 
                ## May need more memory anyways 
                buffer = 0
            del besseljBuffer, normalization 
            
            eig1, eig2, eig3 = eigenvaluefield33(outputfeature_11, outputfeature_12, outputfeature_13,
                                                 outputfeature_22, outputfeature_23, outputfeature_33) 
            
            
            if self.calc_eigenvectors:
                gc.collect() 
                # Sort the eigenvectors with the unique eigenvalues in front 
                eps = 1e-5 
                # If 1 and 2 are the same, rearrange: 
                eig1[xp.abs(eig1-eig2) < eps] = eig3[xp.abs(eig1-eig2) < eps]
                eig3[xp.abs(eig1-eig2) < eps] = eig2[xp.abs(eig1-eig2) < eps]
                # If 1 and 3 are the same, rearrange: 
                eig1[xp.abs(eig1-eig3) < eps] = eig2[xp.abs(eig1-eig3) < eps]
                eig2[xp.abs(eig1-eig3) < eps] = eig3[xp.abs(eig1-eig3) < eps]
                vec1, vec2, vec3 = eigenvectorfield33(eig1, eig2, eig3,
                                                      outputfeature_11, outputfeature_12, outputfeature_13, 
                                                      outputfeature_22, outputfeature_23, outputfeature_33)
            
            del outputfeature_11, outputfeature_12, outputfeature_13, outputfeature_22, outputfeature_23, outputfeature_33    
            
            # Sort the eigenvalues 
            maxe = xp.copy(eig1)
            del eig1 
            mine = xp.copy(maxe)
            mide = maxe + eig2 + eig3
            
            # Also sort the eigenvectors 
            if self.calc_eigenvectors:
                maxe_vec = vec1
                mine_vec = vec1 
                del vec1 
                mide_vec = maxe_vec + vec2 + vec3 
                
            if self.use_absolute:
                maxe[xp.abs(eig2) > xp.abs(maxe)] = eig2[xp.abs(eig2) > xp.abs(maxe)]
                mine[xp.abs(eig2) < xp.abs(mine)] = eig2[xp.abs(eig2) < xp.abs(mine)]
                
                if self.calc_eigenvectors:
                    maxe_vec[xp.abs(eig2) > xp.abs(maxe)] = vec2[xp.abs(eig2) > xp.abs(maxe)]
                    mine_vec[xp.abs(eig2) < xp.abs(mine)] = vec2[xp.abs(eig2) < xp.abs(mine)] 
                    del vec2 
                    
                del eig2 
                maxe[xp.abs(eig3) > xp.abs(maxe)] = eig3[xp.abs(eig3) > xp.abs(maxe)]
                mine[xp.abs(eig3) < xp.abs(mine)] = eig3[xp.abs(eig3) < xp.abs(mine)] 
                
                if self.calc_eigenvectors:
                    maxe_vec[xp.abs(eig3) > xp.abs(maxe)] = vec3[xp.abs(eig3) > xp.abs(maxe)]
                    mine_vec[xp.abs(eig3) < xp.abs(mine)] = vec3[xp.abs(eig3) < xp.abs(mine)]
                    del vec3 
                    
                del eig3 
            else:
                maxe[eig2 > maxe] = eig2[eig2 > maxe]
                mine[eig2 < mine] = eig2[eig2 < mine]
                
                if self.calc_eigenvectors:
                    maxe_vec[eig2 > maxe] = vec2[eig2 > maxe]
                    mine_vec[eig2 < mine] = vec2[eig2 < mine] 
                    del vec2 
                    
                del eig2 
                maxe[eig3 > maxe] = eig3[eig3 > maxe]
                mine[eig3 < mine] = eig3[eig3 < mine]
                
                if self.calc_eigenvectors:
                    maxe_vec[eig3 > maxe] = vec3[eig3 > maxe]
                    mine_vec[eig3 < mine] = vec3[eig3 < mine] 
                    del vec3
                del eig3 
                
            mide = mide - maxe - mine
            if self.calc_eigenvectors:
                mide_vec = mide_vec - maxe_vec - mine_vec
                
            del mine 
            
            # Get the eigenvectors and compute the OFA for each eigenvector 
            if self.calc_eigenvectors:
                # vec1_max[xp.abs(tmpfeature) > xp.abs(output)] = mine_vec[xp.abs(tmpfeature) > xp.abs(output)]
                del mine_vec 
                if self.do_oofofa:
                    ## We need to compute the OFA response for the given eigenvectors 
                
                    ## Write a function for this? 
                    ## ofa_1, ofa_2, ofa_3 = self._compute_ofa(r, radius, imgfft, coords, margin_widths)
                    ofa = 1j / (r*radius) * xp.exp(-2*xp.square(xp.pi*radius*self.sigma)) * xp.sin(2*xp.pi*r*radius)
                    # DO we have to normalize this too?
                    ofa = ofa * (r/xp.sqrt(2*r*self.sigma - self.sigma**2))**self.normalization_type
                    ofa = xp.multiply(ofa,imgfft)
                    
                    if self.response_type in [1,2,3,5]:
                        ofa_mid = xp.real(xp.fft.ifftn((freq_op(coords[0],margin_widths)*mide_vec[:,:,:,0] + \
                                                        freq_op(coords[1],margin_widths)*mide_vec[:,:,:,1] + \
                                                        freq_op(coords[2],margin_widths)*mide_vec[:,:,:,2])*freq_op(ofa,margin_widths)))

                    else:
                        ofa_mid = None 
                        del mide_vec 
                    ofa_max = xp.real(xp.fft.ifftn((freq_op(coords[0],margin_widths)*maxe_vec[:,:,:,0] + \
                                                    freq_op(coords[1],margin_widths)*maxe_vec[:,:,:,1] + \
                                                    freq_op(coords[2],margin_widths)*maxe_vec[:,:,:,2])*freq_op(ofa,margin_widths)))
                
                
            
            if self.response_type == 0:
                tmpfeature = maxe
                if self.do_oofofa:
                    tmpfeature -= ofa_max 
            elif self.response_type == 1:
                tmpfeature = maxe + mide 
                if self.do_oofofa:
                    tmpfeature -= xp.sqrt(xp.square(ofa_mid) + xp.square(ofa_max))
            elif self.response_type == 2:
                tmpfeature = xp.sqrt(xp.maximum(0, maxe*mide))
                if self.do_oofofa:
                    tmpfeature -= xp.sqrt(xp.square(ofa_mid) + xp.square(ofa_max))
            elif self.response_type == 3:
                tmpfeature = xp.sqrt(xp.maximum(0, maxe)*xp.maximum(0, mide))
                if self.do_oofofa:
                    tmpfeature -= xp.sqrt(xp.square(ofa_mid) + xp.square(ofa_max))
            elif self.response_type == 4:
                tmpfeature = xp.maximum(maxe, 0)
                if self.do_oofofa:
                    tmpfeature -= ofa_max 
            elif self.response_type == 5:
                tmpfeature = xp.maximum(maxe+mide, 0)
                if self.do_oofofa:
                    tmpfeature -= xp.sqrt(xp.square(ofa_mid) + xp.square(ofa_max))
            
            
            del mide, maxe
            
            if self.do_oofofa:
                del ofa_max, ofa_mid
            
            output[xp.abs(tmpfeature) > xp.abs(output)] = tmpfeature[xp.abs(tmpfeature) > xp.abs(output)]
            
            if self.calc_eigenvectors:
                ## Store the max. response eigenvectors 
                if self.response_type in [1,2,3,5]:
                    vec2_max[xp.abs(tmpfeature) > xp.abs(output)] = mide_vec[xp.abs(tmpfeature) > xp.abs(output)]
                    del mide_vec 
                
                vec3_max[xp.abs(tmpfeature) > xp.abs(output)] = maxe_vec[xp.abs(tmpfeature) > xp.abs(output)]
                del maxe_vec 
            
            del tmpfeature 
            
            # if self.do_oofofa:
                # # Compute s(x; r,p) for p = the x,y,z unit vectors 
                # ofa_1 = freq_op(xp.real(xp.fft.ifftn(coords[0]*ofa)), margin_widths)
                # ofa_2 = freq_op(xp.real(xp.fft.ifftn(coords[1]*ofa)), margin_widths)
                # ofa_3 = freq_op(xp.real(xp.fft.ifftn(coords[2]*ofa)), margin_widths) 
                # ofa_mag = xp.sqrt(xp.square(ofa_1)+xp.square(ofa_2)+xp.square(ofa_3))
                # del ofa_1, ofa_2, ofa_3, ofa_mag 
            gc.collect()
            
        if self.calc_eigenvectors:
            if self.do_oofofa:
                F = 0
                return output, F #, ofa_1_max, ofa_2_max, ofa_3_max  
            else:
                return output, vec3_max, vec2_max #,vec1_max 
        else:
            return output 
    
    
def ifft_shifted_coor_matrix(dimension, xp=np):
    # The dimension is a vector specifying the size of the returned coordinate matrices. The 
    # number of output argument is equal to the dimensionality of the vector "dimension".
    # xp is either np for numpy or cp for cupy 
    dim = len(dimension)
    p = dimension // 2
    
    result = []
    for i in range(dim):
        a = xp.concatenate((xp.arange(int(p[i]),dimension[i]),xp.arange(int(p[i])))) - p[i]
        reshapepara = xp.ones((dim,), dtype='int')
        reshapepara[i] = dimension[i]
        a = a.reshape((int(reshapepara[0]), int(reshapepara[1]), int(reshapepara[2])))
        repmatpara = dimension.copy() 
        repmatpara[i] = 1
        result.append(xp.tile(a, _cupy_to_list(repmatpara)))
    return result 
        
def ifft_shifted_coordinate(dimension, dimindex, pixelspacing):
    dim = len(dimension)
    p = dimension // 2
    
    a = p[dimindex] + 1
    pass 
       
def _cupy_to_list(array):
    # Deal with the annoying fact that cupy cannot convert any type of slicing to individual ints or floats. 
    return [int(array[i]) for i in range(len(array))]
    
    
    
def freq_op(freq, marginwidth):
    if marginwidth[0] == 0 and marginwidth[1] == 0 and marginwidth[2] == 0:
        result = freq 
    else:
        result = freq[marginwidth[0]:-marginwidth[0],
                      marginwidth[1]:-marginwidth[1],
                      marginwidth[2]:-marginwidth[2]]
    return result       

def eigenvaluefield33(a11,a12,a13,a22,a23,a33):
    xp = cp.get_array_module(a11)
    _epsilon = 1e-50

    a11 = a11.astype('double')
    a12 = a12.astype('double')
    a13 = a13.astype('double')
    a22 = a22.astype('double')
    a23 = a23.astype('double')
    a33 = a33.astype('double')

    b = a11 + _epsilon 
    d = a22 + _epsilon
    j = a33 + _epsilon 

    c = - (a12**2 + a13**2 + a23**2 - b * d - d * j - j * b)
    mul1 = a23**2 * b + a12**2 * j + a13**2 * d
    mul2 = a13 * a12 * a23
    d = - (b * d * j - mul1 + 2 * mul2)
    b = - a11 - a22 - a33 - _epsilon*3
    d += (2 * b**3 - 9 * b * c) / 27
    
    c = (b**2 / 3-c)**3/27
    c = xp.maximum(0, c)
    c = xp.sqrt(c)
    j = c**(1 / 3)
    c += c == 0
    d *= - 1 / 2 / c
    d = xp.clip(d, -1, 1)
    d = xp.real(xp.arccos(d) / 3)
    c = j * np.cos(d)
    d = j * np.sqrt(3) * np.sin(d)
    b *= - 1 / 3
    j = - c - d + b
    d += b - c
    b += 2 * c
    
    eig1 = b.astype('single')
    eig2 = j.astype('single')
    eig3 = d.astype('single')
    return eig1, eig2, eig3
    
    
def eigenvectorfield33(b, j, d, a11, a12, a13, a22, a23, a33):
    # Compute the eigenvectors based on the eigenvalues
    # Based on non-iterative algorithm presented in:
    #
    # "A Robust Eigensolver for 3x3 Symmetric Matrices"
    # David Eberly, Geometric Tools 
    # Redmond WA 98052
    
    
    do_method_2 = True # whether to do method 2 for computing the eigenvectors or not 
    
    xp = cp.get_array_module(b)
    

    vec1 = _compute_first_eig(b, a11, a12, a13, a22, a23, a33)
    
    if not do_method_2:
        # Uncomment the following if we are trying to do by case 
        _Epsilon = 1e-5 
        vec2 = xp.zeros(vec1.shape)
        condition_l = xp.abs(d-j) < _Epsilon
        vec2[~condition_l] = _compute_first_eig(j[~condition_l], a11[~condition_l], a12[~condition_l],
                                                a13[~condition_l], a22[~condition_l], a23[~condition_l], a33[~condition_l])
        vec2[condition_l], vec3 = _compute_orthogonal_complement(vec1) 

    else:
        # Compute the eigenvectors that span the eigenspace for multiplicity 2 
        u, v = _compute_orthogonal_complement(vec1) 
        vec2 = _compute_second_eig(j,u,v,a11,a12,a13,a22,a23,a33)
        vec3 = cross(vec1, vec2) 
    
    return vec1, vec2, vec3
    
def cross(a, b):
    xp = cp.get_array_module(a)
    u = ([a[:,:,:,1]*b[:,:,:,2] - a[:,:,:,2]*b[:,:,:,1],
         a[:,:,:,2]*b[:,:,:,0] - a[:,:,:,0]*b[:,:,:,2],
         a[:,:,:,0]*b[:,:,:,1] - a[:,:,:,1]*b[:,:,:,0]])
    c = xp.stack((u[0],u[1],u[2]),axis=-1)
    return c

def _compute_orthogonal_complement(eigb):
    # Computes an orthogonal vector to a given eigenvector 
    xp = cp.get_array_module(eigb)
    w0 = eigb[:,:,:,0]; w1 = eigb[:,:,:,1]; w2 = eigb[:,:,:,2]
    u0 = xp.zeros(w0.shape); u1 = xp.zeros(w1.shape); u2 = xp.zeros(w2.shape)
    condition = xp.abs(w0) > xp.abs(w1)
    
    invlength_cond = (w0[condition] * w0[condition] + w2[condition]*w2[condition])**-0.5
    u0[condition] = -w2[condition]*invlength_cond 
    u2[condition] = w0[condition]*invlength_cond 
    
    invlength_notcond = (w1[~condition]*w1[~condition]+w2[~condition]*w2[~condition])**-0.5
    u1[~condition] = w2[~condition]*invlength_notcond
    u2[~condition] = -w1[~condition]*invlength_notcond
    
    u = xp.stack((u0,u1,u2), axis=-1)
    u = u / xp.linalg.norm(u) # normalize
    v = cross(eigb, u)
    v = v / xp.linalg.norm(v) # normalize
    
    return u, v
    
def _compute_first_eig(b, a11, a12, a13, a22, a23, a33):
    xp = cp.get_array_module(a33)
    
    a11 = a11.astype('double')
    a12 = a12.astype('double')
    a13 = a13.astype('double')
    a22 = a22.astype('double')
    a23 = a23.astype('double')
    a33 = a33.astype('double')
    
    # Form the rows of the matrix 
    r1 = xp.stack((a11-b,a12,a13), axis=-1)
    r2 = xp.stack((a12,a22-b,a23), axis=-1)
    r3 = xp.stack((a13,a23,a33-b), axis=-1)
    del a11,a12,a13,a22,a23,a33,b
    
    # Take the cross products 
    c1 = cross(r1, r2) 
    c2 = cross(r1, r3)
    del r1
    c3 = cross(r2, r3)
    del r2, r3 
    
    # Take dotproducts 
    d1 = xp.sum(c1*c1, 3)
    d1 = xp.stack((d1,d1,d1),axis=-1)
    d2 = xp.sum(c2*c2, 3)
    d2 = xp.stack((d2,d2,d2),axis=-1)
    d3 = xp.sum(c3*c3, 3)
    d3 = xp.stack((d3,d3,d3),axis=-1)
    
    #Find maximum dot product 
    dmax = xp.copy(d1)
    imax = xp.ones(dmax.shape,dtype='uint8')
    eigb = xp.copy(c1) 
    
    dmax[d2 > dmax] = d2[d2 > dmax]
    imax[d2 > dmax] = 2 
    imax[d3 > dmax] = 3
    eigb[imax==1] = c1[imax==1] / xp.sqrt(d1[imax==1])
    eigb[imax==2] = c2[imax==2] / xp.sqrt(d2[imax==2])
    eigb[imax==3] = c3[imax==3] / xp.sqrt(d3[imax==3])
    
    
    # imax = 1 
    # if d2 > dmax:
        # dmax = d2; imax = 2
    # if d3 > dmax:
        # imax = 3
    # if imax == 1:
        # eigb = c1 / xp.sqrt(d1)
    # elif imax == 2:
        # eigb = c2 / xp.sqrt(d2)
    # elif imax == 3:
        # eigb = c3 / xp.sqrt(d3)
    
    return eigb

def _compute_second_eig(j, u, v, a11, a12, a13, a22, a23, a33):
    xp = cp.get_array_module(a11) 
    
    _Epsilon = 1e-8
    j = j.astype('double')
    
    u0 = u[:,:,:,0]; u1 = u[:,:,:,1]; u2 = u[:,:,:,2]
    v0 = v[:,:,:,0]; v1 = v[:,:,:,1]; v2 = v[:,:,:,2]
    eigj0 = xp.zeros(u0.shape); eigj1 = xp.zeros(u1.shape); eigj2 = xp.zeros(u2.shape)
    Au = xp.stack((a11*u0 + a12*u1 + a13*u2, a12*u0 + a22*u1 + a23*u2, a13*u0 + a23*u1 + a33*u2), axis=-1)
    Av = xp.stack((a11*v0 + a12*v1 + a13*v2, a12*v0 + a22*v1 + a23*v2, a13*v0 + a23*v1 + a33*v2), axis=-1)
    m00 = xp.sum(u*Au, 3) - j
    m01 = xp.sum(u*Av, 3)
    m11 = xp.sum(v*Av,3) - j
    absm00 = xp.abs(m00); absm01 = xp.abs(m01); absm11 = xp.abs(m11)
    
    # if absm00 >= absm11 
    condition = absm00 >= absm11
    maxAbsComp = xp.maximum(absm00, absm01)
    # maxAbsComp = -1*xp.ones(absm00.shape)
    # maxAbsComp[condition] = xp.maximum(absm00[condition], absm01[condition])
    
    # if maxAbsComp > 0 
    condition2 = maxAbsComp > _Epsilon # should account for the first condition as well 
    # Accounts for the else condition 
    else_2 = condition * ~condition2 
    eigj0[else_2] = u0[else_2]
    eigj1[else_2] = u1[else_2]
    eigj2[else_2] = u2[else_2]
    
    # if absm00 >= absm01
    condition3 = absm00 >= absm01
    acc_conds = condition * condition2 * condition3 
    else_conds = condition * condition2 * ~condition3 
    m01[acc_conds] = m01[acc_conds] / m00[acc_conds]
    m00[acc_conds] = 1 / xp.sqrt(1 + m01[acc_conds]*m01[acc_conds])
    m01[acc_conds] = m01[acc_conds] * m00[acc_conds] 
    m00[else_conds] = m00[else_conds] / m01[else_conds] 
    m01[else_conds] = 1 / xp.sqrt(1 + m00[else_conds]*m00[else_conds])
    m00[else_conds] = m00[else_conds] * m01[else_conds] 
    
    # update the eigenvectors 
    eigj0[acc_conds] = m01[acc_conds]*u0[acc_conds] - m00[acc_conds]*v0[acc_conds]
    eigj1[acc_conds] = m01[acc_conds]*u1[acc_conds] - m00[acc_conds]*v1[acc_conds]
    eigj2[acc_conds] = m01[acc_conds]*u2[acc_conds] - m00[acc_conds]*v2[acc_conds]
    eigj0[else_conds] = m01[else_conds]*u0[else_conds] - m00[else_conds]*v0[else_conds]
    eigj1[else_conds] = m01[else_conds]*u1[else_conds] - m00[else_conds]*v1[else_conds]
    eigj2[else_conds] = m01[else_conds]*u2[else_conds] - m00[else_conds]*v2[else_conds]
    
    # Get the other else condition 
    maxAbsComp = xp.maximum(absm11, absm01)
    condition4 = maxAbsComp > _Epsilon 
    else_4 = ~condition * ~condition4
    eigj0[else_4] = u0[else_4]
    eigj1[else_4] = u1[else_4]
    eigj2[else_4] = u2[else_4]
    
    # if absm11 >= absm01 
    condition5 = absm11 >= absm01
    acc_conds = ~condition * condition4 * condition5
    else_conds = ~condition * condition4 * ~condition5 
    m01[acc_conds] = m01[acc_conds] / m11[acc_conds]
    m11[acc_conds] = 1 / xp.sqrt(1 + m01[acc_conds]*m01[acc_conds])
    m01[acc_conds] = m01[acc_conds] * m11[acc_conds] 
    m11[else_conds] = m11[else_conds] / m01[else_conds] 
    m01[else_conds] = 1 / xp.sqrt(1 + m11[else_conds]*m11[else_conds])
    m11[else_conds] = m11[else_conds] * m01[else_conds] 
    
    return xp.stack((eigj0,eigj1,eigj2),axis=-1) 
    
    
    
    
    
    
############### main function: for applying OOF on large image (zarr)
def apply_oof(source_zarr_path, sink_zarr_path, oof_chunks, overlap, radii, 
              slice_range=None, use_cupy=True, num_workers=None, **oof_opts):
    '''
    Applies OOF over arbitrarily large images using chunking.
    
    Inputs:
    source_zarr_path 
    sink_zarr_path - 
    oof_chunks - 
    overlap - int, number of pixels to overlap (in each direction) so that we don't get edge effects and discontinuities. 
    radii - list or array of ints, 
    slice_range - [int,int], the slices to process for OOF in the zarr file. default: None
    use_cupy - bool, if True, then use cupy (GPU sped up OOF). 
    num_workers -
    **oof_opts - see OOF class for the keyword arguments 
    
    Outputs:
    '''
    
    bottom = zarr.open(source_zarr_path,mode='r')
    if slice_range is not None:
        img_size = (*bottom.shape[:2],slice_range[1]-slice_range[0])
    else:
        img_size = bottom.shape
    xr0,yr0,zr0 = (0,0,slice_range[0]) # the absolute first points for each
    bottom_filtered = zarr.open(sink_zarr_path,mode='w',
                                shape=img_size,chunks=oof_chunks,dtype=np.uint16)
                               # compressor=Zstd(level=1)) # add compressor? 
                                               
    # Read a large chunk in and then do smaller chunk processing using OOF 
    global_chunks = get_chunk_coords(img_size, (*bottom.chunks[:2],oof_chunks[2]))

    for coords in global_chunks:
        start = time.time()
        xrl,yrl,zrl = coords
        zrl = [zrl[j] + slice_range[0] for j in [0,1]] # readjust the actual coordinates based on where we are 
        
        print("Starting to read large image chunk")
        star = time.time()
        large_chunk = bottom[xrl[0]:xrl[1],yrl[0]:yrl[1],zrl[0]:zrl[1]]
        print("finished large image chunk read in %f seconds"%(time.time()-star))
            
        local_chunks = get_chunk_coords(large_chunk.shape, oof_chunks)    
        if num_workers is None:
            # Serial version 
            for small_coords in tqdm(local_chunks):
                _apply_oof_serial(bottom_filtered, large_chunk, [xrl,yrl,zrl], 
                                  overlap, radii, use_cupy, oof_opts, small_coords)
        else:
            # Parallel version 
            p = mp.Pool(num_workers)
            
            ## Using the overlap version 
            # f = partial(_apply_oof_serial, bottom_filtered, large_chunk, [xrl,yrl,zrl], 
                                           # overlap, radii, use_cupy, oof_opts)
            # p.map(f, local_chunks)
            
            
            ## Using the non-overlap version 
            # The following is a list of tuples of (img, local_coords) 
            img_chunks = [(large_chunk[c[0][0]:c[0][1],
                                      c[1][0]:c[1][1],
                                      c[2][0]:c[2][1]], c) for c in local_chunks]
            f = partial(_apply_oof_serial, bottom_filtered, [xr0,yr0,zr0], [xrl,yrl,zrl], radii, use_cupy, oof_opts)
            list(tqdm(p.imap(f, img_chunks), total=len(img_chunks)))
            
            
            p.close()
            p.join()
            
        print("Done with chunk x: %d-%d, y: %d-%d, z: %d-%d in %f seconds"% \
              (xrl[0],xrl[1],yrl[0],yrl[1],zrl[0],zrl[1],time.time()-start) )
            
        
## THe following version incorporates overlap     
# def _apply_oof_serial(sink_zarr, img, global_coords, overlap, radii, use_cupy, oof_options, local_coords):
    # # img - either the large chunk of image that's been read into memory, or the small chunk itself (depending on if we use overlap). 
    # # oof_params is a dictionary of the OOF parameters 
    
    # start = time.time() 
    # xr,yr,zr = local_coords
    # xrl,yrl,zrl = global_coords 
    # oof_chunks = sink_zarr.chunks 
    

    # ## Two things to try: passing in large_chunk (memory-loaded image) into the kernel, or forgoing overlap and 
    # ## just passing in an image.comment the following out if we don't intend to use overlap.
    # overlaps = [[overlap]*2]*3
    # if xr[0] < overlap:
        # overlaps[0][0] = xr[0]
    # if xr[1] > img.shape[0] - overlap:
        # overlaps[0][1] = img.shape[0]-xr[1]
    # if yr[0] < overlap:
        # overlaps[1][0] = yr[0]
    # if yr[1] > img.shape[1] - overlap:
        # overlaps[1][1] = img.shape[1]-yr[1]
    # if zr[0] < overlap:
        # overlaps[2][0] = zr[0]
    # if zr[1] > img.shape[2] - overlap:
        # overlaps[2][1] = img.shape[2]-zr[1]
    
    # img = img[xr[0]-overlaps[0][0]:xr[1]+overlaps[0][1],
              # yr[0]-overlaps[1][0]:yr[1]+overlaps[1][1],
              # zr[0]-overlaps[2][0]:zr[1]+overlaps[2][1]] 
    # #####################################################
    
    
    # if use_cupy:
        # radii = cp.asarray(radii)
        # img = cp.asarray(img)
        
    # poof = OOF(img, radii, **oof_options)
    # filtered_img = poof.compute_oof()[overlaps[0][0]:overlaps[0][0]+oof_chunks[0],
                                      # overlaps[1][0]:overlaps[1][0]+oof_chunks[1],
                                      # overlaps[2][0]:overlaps[2][0]+oof_chunks[2]] 
    # if use_cupy:
        # filtered_img = cp.asnumpy(filtered_img)
    
    # # Get globla coordinates and assign to zarr 
    # sink_zarr[xrl[0]+xr[0]:xrl[0]+xr[1],
            # yrl[0]+yr[0]:yrl[0]+yr[1],
            # zrl[0]+zr[0]:zrl[0]+zr[1]] = filtered_img
    # print("Done with chunk x: %d-%d, y: %d-%d, z: %d-%d in %f seconds"% \
           # (xrl[0],xrl[1],yrl[0],yrl[1],zrl[0],zrl[1],time.time()-start) )
           
           
def _apply_oof_serial(sink_zarr, global_start_coords, global_coords, radii, use_cupy, oof_options, p):
    # p[0] - img - small chunk of img 
    # p[1] - local_coords - [xr,yr,zr] of the img in the larger image 
    # oof_options is a dictionary of the OOF parameters 
    # global_start_coords = xr0,yr0,zr0 which are the actual starting points 
    # global_start_coords = xrl,yrl,zrl which are the starting points for this large chunk 
    
    start = time.time() 
    img, local_coords = p
    xr,yr,zr = local_coords
    xrl,yrl,zrl = global_coords 
    xr0,yr0,zr0 = global_start_coords
    oof_chunks = sink_zarr.chunks 
    
    if use_cupy:
        radii = cp.asarray(radii)
        img = cp.asarray(img)
        
    poof = OOF(img, radii, **oof_options)
    filtered_img = poof.compute_oof()
    
    if use_cupy:
        filtered_img = cp.asnumpy(filtered_img)
    filtered_img = filtered_img * (filtered_img > 0) # get rid of negatives 


    # Get globla coordinates and assign to zarr 
    # need to account for the edge cases 
    
    if xr[1]+xrl[0]-xr0 > sink_zarr.shape[0]:
        x_lim = sink_zarr.shape[0]
    else:
        x_lim = xr[1]+xrl[0]-xr0
    if yr[1]+yrl[0]-yr0 > sink_zarr.shape[1]:
        y_lim = sink_zarr.shape[1]
    else:
        y_lim = yr[1]+yrl[0]-yr0
    if zr[1]+zrl[0]-zr0 > sink_zarr.shape[2]:
        z_lim = sink_zarr.shape[2]
    else:
        z_lim = zr[1]+zrl[0]-zr0

    # print('xrl,yrl,zrl:',xrl, yrl, zrl)
    # print('xr,yr,zr:',xr,yr,zr)
    # print('xyz lim:',x_lim, y_lim, z_lim)
    # a = sink_zarr[xr[0]+xrl[0]-xr0:x_lim,
    #           yr[0]+yrl[0]-yr0:y_lim,
    #           zr[0]+zrl[0]-zr0:z_lim]
    # print('a.shape:',a.shape)
    
    
    # Now need to account for cases in which filtered_img has 
    if filtered_img.shape[0] > x_lim - (xr[0]+xrl[0]-xr0):
        filtered_img = filtered_img[:x_lim - (xr[0]+xrl[0]-xr0),:,:]
    if filtered_img.shape[1] > y_lim - (yr[0]+yrl[0]-yr0):
        filtered_img = filtered_img[:,:y_lim - (yr[0]+yrl[0]-yr0),:]
    if filtered_img.shape[2] > z_lim - (zr[0]+zrl[0]-zr0):
        filtered_img = filtered_img[:,:,:z_lim - (zr[0]+zrl[0]-zr0)]
    # print('filtered_img.shape', filtered_img.shape)
    sink_zarr[xr[0]+xrl[0]-xr0:x_lim,
              yr[0]+yrl[0]-yr0:y_lim,
              zr[0]+zrl[0]-zr0:z_lim]= filtered_img
#    print("Done with chunk x: %d-%d, y: %d-%d, z: %d-%d in %f seconds"% \
#           (xr[0],xr[1],yr[0],yr[1],zr[0],zr[1],time.time()-start) )




############### More intelligent way to apply parallelized OOF ####################

def apply_oof_v2(source_zarr_path, sink_zarr_path, radii, 
              slice_range=None, use_cupy=True, num_workers=None, 
              mask_zarr_path=None, downsample_factor=(1,1,1),overlap=(0,0,0),
              **oof_opts):
    '''
    Applies OOF over arbitrarily large images using chunking.
    
    Inputs:
    source_zarr_path - path to the original source image 
    sink_zarr_path - path to write the OOF-filtered chunks
    radii - list or array of ints, the Optimally-oriented flux radii to use 
    slice_range - [int,int], the slices to process for OOF in the zarr file. default: None
    use_cupy - bool, if True, then use cupy (GPU sped up OOF). 
    num_workers - int, number of parallel workers
    **oof_opts - see OOF class for the keyword arguments 

    mask_zarr_path - (optional) str, path to a mask zarr, only if we want to only perform oof on subset of chunks
    downsample_factor - (optional) tuple or list, of the factor by which the mask_zarr is downsampled  
    
    Outputs:
    '''
    
    bottom = zarr.open(source_zarr_path,mode='r')
    if slice_range is not None:
        img_size = (*bottom.shape[:2],slice_range[1]-slice_range[0])
        xr0,yr0,zr0 = (0,0,slice_range[0]) # the absolute first points for each
    else:
        img_size = bottom.shape
        xr0,yr0,zr0 = (0,0,0)

    bottom_filtered = zarr.create(store=zarr.DirectoryStore(sink_zarr_path),
                                shape=img_size,chunks=bottom.chunks,dtype=np.uint16,overwrite=True)
    
    if mask_zarr_path is None:                                           
        coords = np.array(get_chunk_coords(img_size, bottom_filtered.chunks))
    else:
        mask_zarr = zarr.open(mask_zarr_path,mode='r')
        print("Finding relevant chunks...")
        coords = np.array(find_chunk_coords_from_mask(mask_zarr,downsample_factor=downsample_factor,
                                                    original_chunk_size=bottom.chunks,num_workers=num_workers))

    # Sort out the chunks if we're not starting at 0 
    # These are chunks for the original source zarr
    if zr0 != 0:
        coords[:,2,:] += int(np.ceil(bottom_filtered.chunks[2]/zr0)) * bottom_filtered.chunks[2]
        if zr0 % bottom_filtered.chunks[2] != 0:
            num_zslices_leftover = bottom_filtered.chunks[2] - zr0%bottom_filtered.chunks[2]
            coords_add = np.array(get_chunk_coords((*img_size[:2],num_zslices_leftover),
                                                     (*bottom_filtered.chunks[:2],num_zslices_leftover)))
            coords_add[:,2,:] += zr0 
            coords = np.concatenate((coords_add,coords),axis=0)
    if slice_range is not None: #to make sure we don't go over 
        coords[:,2,:][coords[:,2,:] > slice_range[1]] = slice_range[1]

    # Chunks for the sink zarr
    coords_new = coords.copy()
    coords_new[:,2,:] -= zr0 

    # Parallel version 
    print("Starting vessel filter...")
    p = mp.Pool(num_workers)
    coords_total = list(map(lambda q,r:(q,r), coords, coords_new))
    f = partial(_apply_oof_serial_v2, bottom_filtered, bottom, radii, use_cupy, oof_opts, overlap)
    list(tqdm(p.imap(f, coords_total), total=len(coords_total)))
    p.close()
    p.join()

    # Need to ensure that we don't run into blosc decompression errors
    # print("Correcting blosc decompression errors...")
    # correct_blosc_decompression(source_zarr_path, sink_zarr_path, 
    #                             radii, use_cupy, oof_opts)



    
# def _apply_oof_serial_v2(sink_zarr, source_zarr, radii, use_cupy, oof_options, coords):
    
#     source_coords, sink_coords = coords
#     xr,yr,zr = source_coords 
#     x,y,z = sink_coords 

#     img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]

#     if use_cupy:
#         radii = cp.asarray(radii)
#         img = cp.asarray(img)
        
#     poof = OOF(img, radii, **oof_options)
#     filtered_img = poof.compute_oof()
    
#     if use_cupy:
#         filtered_img = cp.asnumpy(filtered_img)
#     filtered_img = filtered_img * (filtered_img > 0) # get rid of negatives 

#     sink_zarr[x[0]:x[1],y[0]:y[1],z[0]:z[1]] = filtered_img 

def _apply_oof_serial_v2(sink_zarr, source_zarr, radii, use_cupy, oof_options, overlap, coords):
    
    source_coords, sink_coords = coords
    xr,yr,zr = source_coords 
    x,y,z = sink_coords 

    # deal with overlap
    x0 = np.maximum(0,xr[0]-overlap[0]); y0 = np.maximum(0,yr[0]-overlap[1]); z0 = np.maximum(0,zr[0]-overlap[2])
    x1 = np.minimum(source_zarr.shape[0],xr[1]+overlap[0]); y1 = np.minimum(source_zarr.shape[1],yr[1]+overlap[1]); z1 = np.minimum(source_zarr.shape[2],zr[1]+overlap[2])

    img = source_zarr[x0:x1,y0:y1,z0:z1]

    if use_cupy:
        radii = cp.asarray(radii)
        img = cp.asarray(img)
        
    poof = OOF(img, radii, **oof_options)
    filtered_img = poof.compute_oof()
    
    if use_cupy:
        filtered_img = cp.asnumpy(filtered_img)
    filtered_img = filtered_img * (filtered_img > 0) # get rid of negatives 

    bbox = [[np.minimum(x0,overlap[0]),np.minimum(np.minimum(x0,overlap[0])+x[1]-x[0],np.minimum(x1-x0,x[1]-x[0]+overlap[0]))],
            [np.minimum(y0,overlap[1]),np.minimum(np.minimum(y0,overlap[1])+y[1]-y[0],np.minimum(y1-y0,y[1]-y[0]+overlap[1]))],
            [np.minimum(z0,overlap[2]),np.minimum(np.minimum(z0,overlap[2])+z[1]-z[0],np.minimum(z1-z0,z[1]-z[0]+overlap[2]))]]

    sink_zarr[x[0]:x[1],y[0]:y[1],z[0]:z[1]] = filtered_img[bbox[0][0]:bbox[0][1],
                                                            bbox[1][0]:bbox[1][1],
                                                            bbox[2][0]:bbox[2][1]]


###### Function to resolve any zarr blosc decompression errors that come up 
# Currently specific to the OOF process since that seems to be where this error occurs #

def correct_blosc_decompression(source_zarr_path, sink_zarr_path, radii, use_cupy, options):
    source_zarr = zarr.open(source_zarr_path, mode='r')
    sink_zarr = zarr.open(sink_zarr_path, mode='r+') # or mode='a'

    # Compute which coordinates are broken 
    # check to see if we will get blosc decompresion error again 
    coords = get_chunk_coords(sink_zarr.shape, sink_zarr.chunks)
    coords_broken = []
    for coord in tqdm(coords):
        xr,yr,zr = coord
        try:
            a = sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
        except:
            coords_broken.append(coord)    

    print("%d chunks experienced errors"%len(coords_broken))
    for coord in coords_broken:
        _apply_oof_serial_v2(sink_zarr, source_zarr, radii, use_cupy=use_cupy, 
                            oof_options=options, coords=(coord,coord))




# Functions that will help us perform OOF (or any other function) in desired coordinate range chunks

# Or we can just look for which chunks have any nonzero values 
def find_chunk_coords_from_mask(mask_zarr, downsample_factor=(1.0,1.0,1.0), original_chunk_size=None,num_workers=24):
    '''
    Based on a (potentially downsampled) mask, we find which chunks have nonzero pixels and thus should
    be included.
    '''
    if original_chunk_size is None:
        original_chunk_size = mask_zarr.chunks 

    # chunk_size to look through this 
    chunk_size = tuple([int(np.round(original_chunk_size[i] / downsample_factor[i])) for i in np.arange(3)])

    coords = get_chunk_coords(mask_zarr.shape, chunk_size)

    # parallelize the search
    p = mp.Pool(num_workers)
    f = partial(_any, mask_zarr, downsample_factor)
    coords_list = list(tqdm(p.imap(f, coords), total=len(coords)))
    p.close()
    p.join()

    coords_final = list(filter(None, coords_list))
    return coords_final 

# kernel function for find_chunk_coords_from_mask
def _any(mask_zarr, downsample_factor, coord):
    xr,yr,zr = coord
    a = mask_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    if np.any(a):
        new_coord = [[int(np.round(coord[i][j]*downsample_factor[i])) for j in [0,1]] for i in [0,1,2] ]
        return new_coord 
    else:
        return None 


# Fixing the error in which OOF doesn't work for certain chunks (redo serially)
def fix_oof_error(source_zarr_path, sink_zarr_path, threshold, 
                  radii, use_cupy, options, num_workers=16, plot=False):
    '''
    Occasionally OOF will not work for a given chunk, determined by if
    the max. of OOF is 0 but the max. of the original chunk is higher than an
    input threshold

    plot - (optional), if True, we plot a 2D scatter of max intensities 
    '''

    z = zarr.open(sink_zarr_path,mode='r+')
    og = zarr.open(source_zarr_path, mode='r')
    coords = get_chunk_coords(z.shape, z.chunks)

    specs = []
    if num_workers <= 1: # serial 
        for coord in tqdm(coords):
            xr,yr,zr = coord
            z_img = z[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
            z_max = z_img.max() 
            if z_max < 1e-5:
                og_img = og[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
                og_max = og_img.max()
                specs.append([coord, z_max,og_max])
    else:
        p = mp.Pool(num_workers)
        f = partial(_chunk_max, z, og)
        coord_max_list = list(tqdm(p.imap(f, coords), total=len(coords)))
        p.close()
        p.join()

        specs = list(filter(None, coord_max_list))

    print("%d chunks to reprocess"%len(specs))
    specs = np.array(specs) 

    if plot:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.scatter(specs[:,2],specs[:,1],alpha=0.35)
        ax.set_xlabel('OG max')
        ax.set_ylabel('OOF max')

    coordsz = list(specs[:,0][specs[:,1] > threshold])

    # Now we need to serially process these 
    for coord in tqdm(coordsz):
        _apply_oof_serial_v2(z, oof, radii, use_cupy=use_cupy, 
                            oof_options=options, coords=(coord,coord))

    # Now verify that it's worked 
    print("Verifying the error correction...")
    for coord in tqdm(coordsz):
        xr,yr,zr = coord
        og_img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
        z_img = sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
        if z_img.max() < 1e-5:
            print("Error still exists in",coord)
    return specs 


def _chunk_max(source_zarr, sink_zarr, coord):
    xr,yr,zr = coord 
    oof_img = sink_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
    oof_max = oof_img.max() 
    if oof_max < 1e-5:
        og_img = source_zarr[xr[0]:xr[1],yr[0]:yr[1],zr[0]:zr[1]]
        og_max = og_img.max() 
        return [coord,oof_max,og_max]
    else:
        return