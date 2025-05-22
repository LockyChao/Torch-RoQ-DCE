# Common helper function
import numpy as np


def crop_center_image_xyz(img, center_pos, patch_shape):
    assert np.all(img.shape[:3] >= patch_shape)

    indices = []
    for n_d in range(len(center_pos)):
        shifting = 0
        min_ind = center_pos[n_d] - patch_shape[n_d] // 2
        max_ind = center_pos[n_d] + patch_shape[n_d] // 2
        if min_ind < 0:
            shifting = -min_ind
        elif max_ind > img.shape[n_d]:
            shifting = img.shape[n_d] - max_ind
        min_ind += shifting
        max_ind += shifting
        indices.append((min_ind, max_ind))

    return img[
           indices[0][0]:indices[0][1],
           indices[1][0]:indices[1][1],
           indices[2][0]:indices[2][1],
           ]

def moving_average(a):
    n=9
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    return np.concatenate(([a[0]],[a[1]],[a[2]],[a[3]],ret,[a[-4]],[a[-3]],[a[-2]],[a[-1]]),axis=0)

def imshow3d(disp_img, ch=0, cmap='gray', is_mask=False, alpha=1.0, **kwargs):
    import matplotlib.pyplot as plt
    disp_slice = disp_img[:, :, disp_img.shape[-2] // 2, ch]

    if not is_mask:
        plt.imshow(disp_slice, cmap=cmap, **kwargs)
    else:
        plt.imshow(disp_slice, cmap=cmap, alpha=np.clip(disp_slice.astype(float), 0.0, alpha), **kwargs)


def mask_roi(dce_img, threshold=0.3, return_diffmap=False):
    """
        Generate a mask to mask the roi of the highest changes in DCE:
        Compute as teh the largest connected components in
        the binary map of >=threshold of the max value of the input map

        input: 4D-DCE image
        output: 3D mask (differenece_map)
    """
    from scipy import ndimage
    diffmap = np.max(np.diff(dce_img, axis=-1), axis=-1)

    # Remove small connected area, just keep the largest one
    # Get the largest connected componet area
    labels, nb = ndimage.label((diffmap >= (threshold * np.max(diffmap))))

    lbl_num, lbl_counts = np.unique(labels, return_counts=True)
    # get the label id
    lbl_id_of_roi = lbl_num[lbl_counts == np.max(lbl_counts[1:])]
    output_image = np.zeros_like(diffmap)
    output_image[labels == lbl_id_of_roi] = 1
    if not return_diffmap:
        return output_image[..., np.newaxis]
    else:
        return output_image[..., np.newaxis], diffmap[..., np.newaxis]


# MATLAB helpers
def reverse_dimensions(in_arr):
    return np.transpose(in_arr, list(reversed(range(len(in_arr.shape)))))


def convert2complex(in_vector):
    """
    if the matlab data == saved with separate real/imagine parts,
    combine together to form complex format

    Might not be necessary but to be compatible with different data formats

    :return: complex output vector
    """

    in_shape = in_vector.shape
    out_vector = in_vector
    try:
        if in_vector.dtype != np.complex128 and in_vector.dtype != np.complex64:
            out_vector = np.array(
                list(map(lambda x: np.complex(x[0], x[1]), in_vector.flatten()))
            )
            out_vector = np.reshape(out_vector, in_shape)
            out_vector = out_vector.astype(np.complex64)
    except IndexError as e:
        print("Convert Complex Array Failed: %s" % e)

    return out_vector


def read_mat_file(mat_filepath):
    import h5py
    from scipy.io import loadmat
    data_dict = {}
    key_list = ['U', 'L', 'Gr', 'Phi', 'Nx', 'Ny', 'Nz', 'cw', 'lm']

    try:
        # New version of matlab file
        with h5py.File(mat_filepath, 'r') as f:
            for k in key_list:
                if k in f:
                    data_dict[k] = convert2complex(
                        reverse_dimensions(np.array(f[k]))
                    )
    except OSError:
        try:
            # Old version of matlab file
            data_dict = loadmat(
                mat_filepath,
                variable_names=key_list
            )

        except IOError as e:
            print(mat_filepath)
            print(e)

    if 'Nx' not in data_dict:
        data_dict['Nx'] = 384
    if 'Ny' not in data_dict:
        data_dict['Ny'] = 276
    if 'Nz' not in data_dict:
        if 'Breast_20190403_shiao_p3t1' in mat_filepath:
            print('p3t1, Nz=152')
            data_dict['Nz'] = 152
        else:
            data_dict['Nz'] = 140
    if 'lm' not in data_dict:
        data_dict['lm'] = [0]
    if 'L' not in data_dict:
        data_dict['L'] = 6
    if 'cw' not in data_dict:
        data_dict['cw'] = 0.00041136
    return data_dict
def smart_mkdir(_dir):
    from pathlib import Path
    Path(_dir).mkdir(parents=True, exist_ok=True)

    
import os
import numpy as np
from numpy import ones, kron, mean, eye, hstack, dot, tile
from numpy.linalg import pinv

def icc(Y, icc_type='ICC(2,1)'):
    ''' Calculate intraclass correlation coefficient

    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

    Args:
        Y: The data Y are entered as a 'table' ie. subjects are in rows and repeated
            measures in columns
        icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)) 
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    '''

    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k-1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc  # / n (without n in SPSS results)

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == 'icc1':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'ICC(2,1)' or icc_type == 'ICC(2,k)':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        if icc_type == 'ICC(2,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'ICC(3,1)' or icc_type == 'ICC(3,k)':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        if icc_type == 'ICC(3,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

    return ICC
def MyPlot(x,y,label,limit,savepath,prefix,mode='debug'):
    """
    plot PK parameters with label, same x-y limit, plot regression line
    return r_value, ICC and CoV
    """
    import matplotlib.pyplot as plt
    import scipy
    from scipy.stats import variation 
    import time 
    """
    R value calculation
    """
    timenow = time.time()
    slope, intercept, R_val, p_value, std_err = scipy.stats.linregress(x,y)
    
    """
    ICC 
    """
    
    _x = x.flatten().reshape(len(x),1)
    _y = y.flatten().reshape(len(y),1)
    icc_data = np.concatenate((_x,_y),axis=1)
    print(icc_data.shape)
    if mode != 'debug':
        icc_val = icc(icc_data,icc_type='ICC(3,1)')
    
#     print(icc(icc_data,icc_type='ICC(2,1)'),icc(icc_data,icc_type='ICC(2,k)'),
#           icc(icc_data,icc_type='ICC(3,1)'),icc(icc_data,icc_type='ICC(3,k)'))

    """
    CoV
    """
    var = variation(icc_data, axis=1)
    CoV_val = np.mean(var,axis=0)*100

    if mode !='value_only':
        fig = plt.figure(figsize=(5,5),facecolor='w',edgecolor='k',dpi=300)
        plt.scatter(x,y,label='data')
        tmp_x = np.linspace(0,limit,100)
        plt.plot(tmp_x,slope*tmp_x+intercept,linestyle='dashed',color='red',label='regression')
        plt.plot(tmp_x,tmp_x,linestyle='solid',color='black',label='y=x')
        plt.xlim([0,limit])
        plt.ylim([0,limit])
        plt.xlabel(label[0])
        plt.ylabel(label[1])
        if mode != 'debug':
            plt.text(limit*0.1,limit*0.8,'R2 = %2f\nICC = %2f\nCoV = %2f%s'%(R_val**2,icc_val,CoV_val,'%'),fontweight='bold')
        else:
            plt.text(limit*0.1,limit*0.8,'R2 = %2f\nCoV = %2f%s'%(R_val**2,CoV_val,'%'),fontweight='bold')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(savepath,'%s.png'%prefix))
    return (R_val**2,icc_val,CoV_val)
    
def bland_altman_plot(data1, data2, *args, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = (data1 - data2)                 # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    
    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean',fontsize=16)
    plt.ylabel('Difference',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
def bland_altman_plot_4_categories(data1, data2, *args, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    
    (pan_1,tum_1,nt_1,ps_1,all_1) = data1
    (pan_2,tum_2,nt_2,ps_2,all_2) = data2
    all_1     = np.asarray(all_1)
    all_2     = np.asarray(all_2)
    mean      = np.mean([all_1, all_2], axis=0)
    diff      = (all_1 - all_2)                # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
#     plt.scatter(mean, diff, *args, **kwargs)
    plt.scatter(np.mean([pan_1, pan_2], axis=0),(pan_1 - pan_2),c='r' )
    plt.scatter(np.mean([tum_1, tum_2], axis=0),(tum_1 - tum_2),c='b' )
    plt.scatter(np.mean([nt_1, nt_2], axis=0),(nt_1 - nt_2),c='g' )
    plt.scatter(np.mean([ps_1, ps_2], axis=0),(ps_1 - ps_2),c='y' )
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean',fontsize=16)
    plt.ylabel('Difference',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
def bland_altman_plot_2_categories(data1, data2, *args, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    
    (tum_1,nt_1,all_1) = data1
    (tum_2,nt_2,all_2) = data2
    all_1     = np.asarray(all_1)
    all_2     = np.asarray(all_2)
    mean      = np.mean([all_1, all_2], axis=0)
    diff      = (all_1 - all_2)                # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff)            # Standard deviation of the difference

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
#     plt.scatter(mean, diff, *args, **kwargs)
    plt.scatter(np.mean([tum_1, tum_2], axis=0),(tum_1 - tum_2),c='b' )
    plt.scatter(np.mean([nt_1, nt_2], axis=0),(nt_1 - nt_2),c='g' )
#     plt.scatter(np.mean([ps_1, ps_2], axis=0),(ps_1 - ps_2),c='y' )
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean',fontsize=16)
    plt.ylabel('Difference',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)  
def bland_altman_plot_2_categories_rel(data1, data2, *args, **kwargs):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    
    (tum_1,nt_1,all_1) = data1
    (tum_2,nt_2,all_2) = data2
    all_1     = np.asarray(all_1)
    all_2     = np.asarray(all_2)
    mean      = np.mean([all_1, all_2], axis=0)
    diff      = (all_1 - all_2)                # Difference between data1 and data2
    md        = np.mean(diff)/mean                   # Mean of the difference
    sd        = np.std(diff)/mean            # Standard deviation of the difference

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
#     plt.scatter(mean, diff, *args, **kwargs)
    plt.scatter(np.mean([tum_1, tum_2], axis=0),(tum_1 - tum_2)/np.mean([tum_1, tum_2], axis=0),c='b' )
    plt.scatter(np.mean([nt_1, nt_2], axis=0),(nt_1 - nt_2)/np.mean([nt_1, nt_2], axis=0),c='g' )
#     plt.scatter(np.mean([ps_1, ps_2], axis=0),(ps_1 - ps_2),c='y' )
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean',fontsize=16)
    plt.ylabel('Difference',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)  
def plot_rand_curve(x,num):
    """
    x: (N,N_len)
    n: num of curves 
    """
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(5*5.75,5),facecolor='w',edgecolor='k',dpi=300)
    test_ind = np.linspace(0,x.shape[0]-1,num=num,dtype=np.int16)
    plt.plot(x[test_ind,:])
    for j in range(num):
        pix_ind = test_ind[j]
        plt.subplot(1,num,j+1)
        plt.title('Pixel:%d' % (pix_ind))
        
        plt.plot(x[pix_ind,:],label='Prediction',linewidth=2,color='blue')
#         plt.legend()
def plot_rand_result(x_inp,x_tar,x_out,num):
    """
    x: (N,N_len)
    n: num of curves 
    """
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(5*5.75,5),facecolor='w',edgecolor='k',dpi=300)
    test_ind = np.linspace(0,x_inp.shape[0]-1,num=num,dtype=np.int16)

    for j in range(num):
        pix_ind = test_ind[j]
        plt.subplot(1,num,j+1)
        plt.title('Pixel:%d' % (pix_ind))
        plt.plot(x_inp[pix_ind,:],label='input',linewidth=2)
        plt.plot(x_tar[pix_ind,:],label='target',linewidth=2)
        plt.plot(x_out[pix_ind,:],label='output',linewidth=2)
        plt.legend()
        
def plot_rand_result_AIF(x_inp,x_tar,x_out,num):
    """
    input,AIF,output
    x: (N,N_len)
    n: num of curves 
    """
    import matplotlib.pyplot as plt
    fig1 = plt.figure(figsize=(5*5.75,5),facecolor='w',edgecolor='k',dpi=300)
    test_ind = np.linspace(0,x_inp.shape[0]-1,num=num,dtype=np.int16)

    for j in range(num):
        pix_ind = test_ind[j]
        plt.subplot(1,num,j+1)
        plt.title('Pixel:%d' % (pix_ind))
        plt.plot(x_inp[pix_ind,:],label='input',linewidth=2)
        plt.plot(x_tar[pix_ind,:],label='AIF',linewidth=2)
        plt.plot(x_out[pix_ind,:],label='output',linewidth=2)
        plt.legend()
def plot_single_result(x_inp,x_tar,x_out):
    """
    x: (1,N_len)
    """
    import matplotlib.pyplot as plt
    fig1 = plt.figure(facecolor='w',edgecolor='k')
    
    plt.title('single pixel')
    plt.plot(x_inp.reshape(-1,1),label='input',linewidth=2)
    plt.plot(x_tar.reshape(-1,1),label='target',linewidth=2)
    plt.plot(x_out.reshape(-1,1),label='output',linewidth=2)
    plt.legend()
def get_train_test_index(CVSP,NUM_FOLDS):
    lis = list(range(NUM_FOLDS))
    lis.remove(CVSP)
    return lis, CVSP

def read_mat_to_variable(matfile):
    from scipy.io import loadmat
    # h5file = loadmat(os.path.join(_dir,"TK_all_fit.mat"))
    h5file = loadmat(matfile)
    
    for key in h5file.keys():
        print(key)
        exec("%s=h5file['%s']"%(str(key),str(key)))