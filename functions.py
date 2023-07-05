import numpy as np
import matplotlib.pyplot as plt

#from matplotlib import cm
from cmcrameri import cm
import matplotlib.colors as colors

me = (0.511*1e6)/(299792458 *1e10)**2  # eV s^2 Angstrom^-2
hbar = 6.582*1e-16   # eV s

#cmp = plt.get_cmap('bone')
#cmp = cmp.reversed()

#cmp = plt.get_cmap('coolwarm')
cmp = cm.devon
cmp = cmp.reversed()

from scipy.io import savemat
from scipy.io import loadmat


import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interactive

import lmfit as lmfit

from lmfit.models import GaussianModel, LorentzianModel, VoigtModel, LinearModel, ConstantModel, StepModel

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
import scipy as scipy

import tqdm


def import_itx(data_path, row_split):  # Based on the code by Pranab Das (GitHub: @pranabdas)
    '''
    import_itx('itx_file.itx')
    Imports Igor text data. This function support reading/importing 1-, 2-, and
    3-dimensional data waves.
    '''
    import numpy as np
    contents = open(data_path, "r").readlines()
    
    def to_int(x):
        return int(float(x))

    # 1D data file
    if (contents[1][:8]=='WAVES/D\t' or contents[1][:8]=='WAVES/D '):
        data = []
        for ii in range(len(contents)):
            if contents[ii+3] == 'END\n':
                break
            data.append(float(contents[ii + 3].lstrip('\t').rstrip('\n')))
            dimsize = ii+1

        temp = contents[dimsize+4].split(';')[0]
        dimoffset = float(temp[temp.find('x')+1:].split(',')[0])
        dimdelta = float(temp[temp.find('x')+1:].split(',')[1])
        x = np.linspace(dimoffset, (dimoffset + dimdelta*(dimsize - 1)), dimsize)

        return data, x

    # Higher dimensional data file
    else:
        dimsize = (contents[1][contents[1].find('(')+1 : contents[1].find(')')]).split(',')
        dimsize = list(map(to_int, dimsize))

        # 2-dimensional data
        if (len(dimsize)==2):        # !!!!!!!!!!!!!!!!!!
            data = np.ndarray((dimsize[0], dimsize[1]))
            for ii in range(dimsize[0]):
                data_row = contents[ii+3][contents[ii+3].find('\t')+1:contents[ii+3].find('\n')]
                # !!!!!!!!!!!
                data_row = data_row.split(row_split)
               
                data[ii, :] = np.asarray(list(map(float, data_row)))

            temp = contents[4+dimsize[0]].split(';')
            x_offset = float(temp[0][temp[0].find('x')+1:temp[0].find('"')].split(',')[:-1][0])
            x_delta = float(temp[0][temp[0].find('x')+1:temp[0].find('"')].split(',')[:-1][1])
            y_offset = float(temp[1][temp[1].find('y')+1:temp[1].find('"')].split(',')[:-1][0])
            y_delta = float(temp[1][temp[1].find('y')+1:temp[1].find('"')].split(',')[:-1][1])

            x_end = x_offset + x_delta*(dimsize[0]-1)
            x = np.linspace(x_offset, x_end, dimsize[0])

            y_end = y_offset + y_delta*(dimsize[1]-1)
            y = np.linspace(y_offset, y_end, dimsize[1])

            return data, x, y

        # 3-dimensional data
        elif (len(dimsize)==3):
            data = np.ndarray((dimsize[0]*dimsize[2], dimsize[1]))
            for ii in range(dimsize[0]*dimsize[2]):
                data_row = contents[ii+3][contents[ii+3].find('\t')+1:contents[ii+3].find('\n')]
                data_row = data_row.split(row_split)
                data[ii, :] = np.asarray(list(map(float, data_row)))

            data = data.reshape(dimsize[0]*dimsize[1]*dimsize[2])
            data = data.reshape((dimsize[1], dimsize[0], dimsize[2]), order='F')
            data = np.transpose(data, (1, 0, 2))

            temp = contents[4+dimsize[0]*dimsize[2]].split(';')
            x_offset = float(temp[0][temp[0].find('x')+1:temp[0].find('"')].split(',')[:-1][0])
            x_delta = float(temp[0][temp[0].find('x')+1:temp[0].find('"')].split(',')[:-1][1])
            y_offset = float(temp[1][temp[1].find('y')+1:temp[1].find('"')].split(',')[:-1][0])
            y_delta = float(temp[1][temp[1].find('y')+1:temp[1].find('"')].split(',')[:-1][1])
            z_offset = float(temp[2][temp[2].find('z')+1:temp[2].find('"')].split(',')[:-1][0])
            z_delta = float(temp[2][temp[2].find('z')+1:temp[2].find('"')].split(',')[:-1][1])

            x_end = x_offset + x_delta*(dimsize[0]-1)
            x = np.linspace(x_offset, x_end, dimsize[0])

            y_end = y_offset + y_delta*(dimsize[1]-1)
            y = np.linspace(y_offset, y_end, dimsize[1])

            z_end = z_offset + z_delta*(dimsize[2]-1)
            z = np.linspace(z_offset, z_end, dimsize[2])

            return data, x, y, z




def find_value_index(matrix, y):
    x = np.array([])
    x = np.append(x, y)
    index = np.zeros(len(x))
    for i in range(len(x)):
        difference_array = np.abs( matrix - x[i] )
        index[i] = difference_array.argmin()
    index = index.astype(int)
    return index




def Raw_data_plot(data, data_Y, data_Ek, hv, sample_name, output_filepath):
    fig = plt.figure(figsize=(4, 4))

    control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )
    if_axhline = widgets.Checkbox( value=True, description='Position For Ek Maximum' )

    button = widgets.Button(description="Savefig")

    ax = fig.add_subplot()
    ax.pcolormesh( data_Y[:,control_hv_index.value], data_Ek[:,control_hv_index.value], np.transpose(data[:,:,control_hv_index.value]), cmap=cmp )

    plt.close()
    
    def hv_slice_plot(control_index, if_axhline):
        ax.clear()
        ax.pcolormesh( data_Y[:,control_index], data_Ek[:,control_index], np.transpose(data[:,:,control_index]), cmap=cmp, zorder=0 )
        if if_axhline:
            ax.axhline(y=data_Ek[-1,control_index], linestyle='--', linewidth=1, zorder=1 )
        ax.set_title("Z = "+str(round(hv[control_index],2)) )
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        ax.set_ylim(top=data_Ek[-1,control_index]+0.1)
    
        display(fig)
    

    interactive_plot = interactive(hv_slice_plot, control_index=control_hv_index, if_axhline=if_axhline)
    output = interactive_plot.children[-1]

    output.layout.height = '400px'

    ui1 = widgets.HBox( [control_hv_index, if_axhline, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_Ek_Y_'+str(hv[control_hv_index.value])+'eV.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)
    
    return fig, ax
    
    
    
    

def Fermi_edge_fit_plot(data, eb, hv, index_range, T, fermi_edge_center, fermi_edge_amplitude, fermi_edge_const, fermi_edge_slope, sample_name, output_filepath):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    
    def Fermi_Dirac(x, amplitude, center, T, const, slope):
        test = np.zeros(len(x))
        for i in range(len(x)):
            test[i] = np.minimum( 1, np.maximum((x[i]-center)/0.0001,0) )
    
        return amplitude * 1 / (np.exp( -(x-center)/(8.617e-5 * T) ) + 1) + const + slope* test * (x-center)

    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(1100*px, 350*px))
    
    x = eb[index_range[0]:index_range[1]]
    x_fit = np.linspace(x[0], x[-1], num=100, endpoint=True)

    control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )

    button = widgets.Button(description="Savefig")

    Eb_0_index = find_value_index(eb, 0)[0]
    curve_data = ax1.plot( x, data[index_range[0]:index_range[1],control_hv_index.value], zorder=0, label='Data', marker='+' )
    curve_fit = ax1.plot( x_fit,  Fermi_Dirac(
                                x_fit, fermi_edge_amplitude[control_hv_index.value],
                                fermi_edge_center[control_hv_index.value], T, fermi_edge_const[control_hv_index.value], fermi_edge_slope[control_hv_index.value]
                                ), zorder=1, label='Fit')  
    ax1.set_xlabel("Binding Energy (eV)")
    #ax1.set_yticks([])
    ax1.set_xlim(np.min(x), np.max(x))
    ax1.legend(loc=2)
    
    ax2.plot(hv, fermi_edge_center, zorder=0)
    scatter = ax2.scatter(hv[control_hv_index.value], fermi_edge_center[control_hv_index.value], color='r', zorder=1)
    ax2.set_xlabel("hv (eV)")
    ax2.set_ylabel("Offset (eV)")

    plt.close()
    
    def hv_slice_plot(control_index):
        curve_data[0].set_ydata(data[index_range[0]:index_range[1],control_index])
        curve_fit[0].set_ydata(Fermi_Dirac(
                                x_fit, fermi_edge_amplitude[control_index],
                                fermi_edge_center[control_index], T, fermi_edge_const[control_index], fermi_edge_slope[control_index]
                                ))
        ax1.relim()
        ax1.autoscale(axis='y')
        ax1.set_title("hv = "+str(round(hv[control_index],2))+" eV")
        
        scatter.set_offsets([hv[control_index], fermi_edge_center[control_index]])
        ax2.set_title('Offset = '+str(round(fermi_edge_center[control_index],3))+ ' eV' )
    
        display(fig)
    

    interactive_plot = interactive(hv_slice_plot, control_index=control_hv_index)
    output = interactive_plot.children[-1]

    output.layout.height = '360px'

    ui1 = widgets.HBox( [control_hv_index, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_Fermi_edge_fit_'+str(hv[control_hv_index.value])+'eV.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)
    
    return fig, ax1, ax2
    
    
    

    
def z_slice(data, x, y, z, sample_name, output_filepath, figsize=(5,5)):
    fig = plt.figure(figsize=figsize)

    z_index_0 = find_value_index(z, 0)[0]

    control_index = widgets.IntSlider( min=0, max=len(z)-1, step=1, value=z_index_0, description='Z Index', continuous_update=True )
    control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )
    #widgets.SelectionRangeSlider( options=options, index=(0, 1), description='Months (2015)', disabled=False )
    if_axhline = widgets.Checkbox( value=True, description='Fermi Edge', layout=widgets.Layout(width='10%'), indent=False )
    
    control_bar_len = widgets.FloatSlider( min=0, max=10, step=0.1, value=1, description=r'Bar len', continuous_update=True )
    control_bar_center = widgets.FloatSlider( min=-8, max=8, step=0.1, value=0, description=r'Offset', continuous_update=True )
    if_bar = widgets.Checkbox( value=False, description='Show bar', layout=widgets.Layout(width='10%'), indent=False )

    button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False )

    ax = fig.add_subplot()
    mesh = ax.pcolormesh( x, y, np.transpose(data[:,:,control_index.value]), cmap=cmp, zorder=0 )
    hline = ax.axhline(y=0, linestyle='--', linewidth=1, zorder=1, color=[0.8, 0.2, 0.2] )
    
    bar = ax.plot([control_bar_center.value-control_bar_len.value/2.0, control_bar_center.value + control_bar_len.value/2.0],[0, 0], color='r')
    center_line = ax.axvline(control_bar_center.value, color='r', linestyle='--')
        
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.close()
    
    def z_slice_plot(control_index, control_gamma, if_axhline, control_bar_len, control_bar_center, if_bar):
        mesh.set_array( np.transpose(data[:,:,control_index]) )
        mesh.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[:,:,control_index]), vmax=np.max(data[:,:,control_index])) )
        bar[0].set_xdata([control_bar_center-control_bar_len/2.0, control_bar_center + control_bar_len/2.0])
        center_line.set_xdata(control_bar_center)
        if if_axhline:
            hline.set_alpha(1)
        else:
            hline.set_alpha(0)

        if if_bar:
            bar[0].set_alpha(1)
            center_line.set_alpha(1)
        else:
            bar[0].set_alpha(0)
            center_line.set_alpha(0)
        
        ax.set_title( "Z = "+str(round(z[control_index],2)) )
                 
        display(fig)
    

    interactive_plot = interactive(z_slice_plot, control_index=control_index, control_gamma=control_gamma, if_axhline=if_axhline, 
                                   control_bar_len=control_bar_len, control_bar_center=control_bar_center, if_bar=if_bar)
    output = interactive_plot.children[-1]

    output.layout.height = str(100*figsize[1] + 10) + 'px'

    ui1 = widgets.HBox( [control_index, control_gamma, if_axhline, button] )
    ui2 = widgets.HBox( [control_bar_center, control_bar_len, if_bar] )
    ui = widgets.VBox( [ui1, ui2, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1] + '_X_Y_' + str(z[control_index.value]) + '.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)
    
    return fig, ax
    
    
    

    
def x_slice(data, x, y, z, sample_name, bar_length, output_filepath, figsize=(5,5)):
    fig = plt.figure(figsize=figsize)
    
    x_index_0 = find_value_index(x, 0)[0]

    control_index = widgets.IntSlider( min=0, max=len(x)-1, step=1, value=x_index_0, description='X Index', continuous_update=True )
    control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )
    if_axhline = widgets.Checkbox( value=True, description='Fermi Edge', layout=widgets.Layout(width='10%'), indent=False )
    
    if (bar_length != '') & (bar_length != 0):
        if_bar = widgets.Checkbox( value=True, description='BZ len bar', layout=widgets.Layout(width='10%'), indent=False )

    button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False)

    ax = fig.add_subplot()
    mesh = ax.pcolormesh( z, y, data[control_index.value,:,:], cmap=cmp, zorder=0 )
    hline = ax.axhline(y=0, linestyle='--', linewidth=1, zorder=1)
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    
    if (bar_length != '') & (bar_length != 0):
        barline = ax.plot( np.linspace(z[0], z[0]+bar_length, num=10, endpoint=True), -0.07*np.ones(10), zorder=3 )

    plt.close()
    
    def x_slice_plot_barless(control_index, if_axhline, control_gamma):
        mesh.set_array( data[control_index,:,:] )
        mesh.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[control_index,:,:]), vmax=np.max(data[control_index,:,:])) )
        if if_axhline:
            hline.set_alpha(1)
        else:
            hline.set_alpha(0)
            
        ax.set_title( "X = "+str(round(x[control_index],2)) )
    
        display(fig)
    
    def x_slice_plot_bar(control_index, if_axhline, if_bar, control_gamma):
        mesh.set_array( data[control_index,:,:] )
        mesh.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[control_index,:,:]), vmax=np.max(data[control_index,:,:])) )
        if if_axhline:
            hline.set_alpha(1)
        else:
            hline.set_alpha(0)
            
        if if_bar:
            barline[0].set_alpha(1)
        else:
            barline[0].set_alpha(0)
            
        ax.set_title( "X = "+str(round(x[control_index],2)) )
    
        display(fig)
    
    if (bar_length != '') & (bar_length != 0):
        interactive_plot = interactive(x_slice_plot_bar, control_index=control_index, control_gamma=control_gamma, if_bar=if_bar, if_axhline=if_axhline)
    else:
         interactive_plot = interactive(x_slice_plot_barless, control_index=control_index, control_gamma=control_gamma, if_axhline=if_axhline)
    output = interactive_plot.children[-1]

    output.layout.height = str(100*figsize[1] + 10) + 'px'
    
    if (bar_length != '') & (bar_length != 0):
        ui1 = widgets.HBox( [control_index, control_gamma, if_axhline, if_bar, button] )
    else:
        ui1 = widgets.HBox( [control_index, control_gamma, if_axhline, button] )

    ui = widgets.VBox( [ui1, output] )
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1] + str(round(x[control_index.value],2)) + '_Y_Z_' + '.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)
    
    return fig, ax
    
    
    
    
    
    
def y_slice(data, x, y, z, sample_name, if_swap, output_filepath, figsize=(5, 5)):

    if if_swap == True:
        temp = x
        x = z; z = temp
        data = np.swapaxes(data, 0, 2)
         
    fig = plt.figure(figsize=figsize)

    y_index_0 = find_value_index(y, 0)[0]

    control_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=y_index_0, description='Y Index', continuous_update=True )
    control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )

    button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False)

    ax = fig.add_subplot()
    mesh = ax.pcolormesh( x, z, np.transpose(data[:,control_index.value,:]), cmap=cmp, zorder=0 )
    
    if if_swap == True:
        ax.set_xlabel('Z')
        ax.set_ylabel('X')
    else:
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
    
    plt.close()
    
    def y_slice_plot(control_index, control_gamma):
        mesh.set_array( np.transpose(data[:,control_index,:]) )
        mesh.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[:,control_index,:]), vmax=np.max(data[:,control_index,:])) )
        
        ax.set_title( "Y = "+str(round(y[control_index],2)) )
    
        display(fig)
    

    interactive_plot = interactive(y_slice_plot, control_index=control_index, control_gamma=control_gamma)
    output = interactive_plot.children[-1]

    output.layout.height = str(100*figsize[1] + 10) + 'px'

    ui1 = widgets.HBox( [control_index, control_gamma, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_X_' + '_' + str(round(y[control_index.value],2)) + '_Z.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)    
    
    return fig, ax

    
    
    
    
  
    
def EDC_angle_sum(data, Y, eb, hv, eb_range, sample_name, output_filepath):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    
    fig, ax = plt.subplots(1, 1, figsize=(600*px, 250*px))
    
    #plt.rcParams['axes.autolimit_mode'] = 'data'
    
    eb_index_range = find_value_index(eb, eb_range)
    
    x = eb[eb_index_range[1]:eb_index_range[0]]

    control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )
    if_axvline = widgets.Checkbox( value=True, description='Fermi Edge' )

    button = widgets.Button(description="Savefig")

    Eb_0_index = find_value_index(eb, 0)[0]
    curve_data = ax.plot( x, data[eb_index_range[1]:eb_index_range[0],control_hv_index.value], zorder=1, label='Data', marker='+' ) 
    vline = ax.axvline(x=0, linestyle='--', linewidth=1, zorder=0, color='orange')
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_yticks([])
    ax.set_xlim(np.min(x), np.max(x))
    
    plt.close()
    
    def EDC_plot(control_index, if_axvline):
        curve_data[0].set_ydata(data[eb_index_range[1]:eb_index_range[0],control_index])
        ax.relim()
        ax.autoscale(axis='y')
        ax.set_title("hv = "+str(round(hv[control_index],2))+" eV")
        if if_axvline:
            vline.set_alpha(1)
        else:
            vline.set_alpha(0)  
        
        display(fig)
    

    interactive_plot = interactive(EDC_plot, control_index=control_hv_index, if_axvline=if_axvline)
    output = interactive_plot.children[-1]

    output.layout.height = '280px'

    ui1 = widgets.HBox( [control_hv_index, if_axvline, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_EDC_'+str(hv[control_hv_index.value])+'eV.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)
    
    return fig, ax
    
    
    

    
    
def EDC(data, Y, eb, hv, eb_range, sample_name, output_filepath):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    
    fig, ax = plt.subplots(1, 1, figsize=(1000*px, 300*px))
    
    eb_index_range = find_value_index(eb, eb_range)
    angle_index_0 = find_value_index(Y, 0)[0]
    
    x = eb[eb_index_range[1]:eb_index_range[0]]

    control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )
    control_Y_index = widgets.IntSlider( min=0, max=len(Y)-1, step=1, value=angle_index_0, description='Y Index', continuous_update=True )
    if_axvline = widgets.Checkbox( value=True, description='Fermi Edge' )

    button = widgets.Button(description="Savefig")

    Eb_0_index = find_value_index(eb, 0)[0]
    curve_data = ax.plot( x, data[control_Y_index.value, eb_index_range[1]:eb_index_range[0],control_hv_index.value], zorder=1, label='Data', marker='+' ) 
    vline = ax.axvline(x=0, linestyle='--', linewidth=1, zorder=0, color='orange')
    ax.set_xlabel("Binding Energy (eV)")
    ax.set_yticks([])
    ax.set_xlim(np.min(x), np.max(x))
    
    plt.close()
    
    def EDC_plot(control_hv_index, control_Y_index, if_axvline):
        curve_data[0].set_ydata( data[control_Y_index, eb_index_range[1]:eb_index_range[0],control_hv_index])
        ax.relim()
        ax.autoscale(axis='y')
        ax.set_title('hv = ' + str(round(hv[control_hv_index],2)) + ' eV' + ', Y = ' + str(round(Y[control_Y_index],2)) + ' Deg')
        if if_axvline:
            vline.set_alpha(1)
        else:
            vline.set_alpha(0)  
        
        display(fig)
    

    interactive_plot = interactive(EDC_plot, control_hv_index=control_hv_index, control_Y_index=control_Y_index, if_axvline=if_axvline)
    output = interactive_plot.children[-1]

    output.layout.height = '320px'

    ui1 = widgets.HBox( [control_hv_index, control_Y_index, if_axvline, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_EDC_'+str(hv[control_hv_index.value])+'eV_'+str(round(Y[control_Y_index.value],2))+'deg.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
        
    button.on_click(on_button_clicked)
    
    return fig, ax
    
    
    
    
    
def Tune_V0(data, x, y, hv, BZ_len, x_name, z_name, sample_name, output_filepath, E_work):
    fig = plt.figure(figsize=(8, 4))

    control_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=round(0.9*len(y)), description='Eb Index', continuous_update=True )
    control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )
    control_V0 = widgets.FloatSlider( min=-30, max=30, step=0.5, value=12, description=r'V0', continuous_update=True )

    button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False)
    
    z = 1/hbar * np.sqrt( 2*me* (hv - 0 - E_work + control_V0.value) ) 

    ax = fig.add_subplot()
    
    mesh = ax.pcolormesh( z, x, data[:,control_index.value,:], cmap=cmp, zorder=0 )
    
    ax.grid(True)
    ax.set_xlim([min(z)-0.5*BZ_len, max(z)+0.5*BZ_len])
    
    hv_cal = hbar**2 * np.arange( 0, BZ_len * 30, BZ_len )**2 / (2*me) + E_work - control_V0.value
    hv_cal = [str(round(x,2)) for x in hv_cal]
    
    x_ticks = np.arange( 0, BZ_len * 30, BZ_len )
    x_ticks = [str(round(x,2)) for x in x_ticks]
    
    new_labels = [ '\n\n'.join(x) for x in zip( x_ticks, hv_cal  ) ]
    
    ax.set_xticks(np.arange( 0, BZ_len * 30, BZ_len ), new_labels)
    
    
    if x_name == 'Y':
        ax.set_ylabel('Y (Deg)')
    elif x_name == 'kp':
        ax.set_ylabel(r'$k_\parallel$ ($\AA^{-1}$)')
    else:
        print( 'y_name error' )
        raise
    
    if z_name == 'hv':
        ax.set_xlabel('hv (eV)')
    elif z_name == 'kz':
        ax.set_xlabel(r'$k_z$ ($\AA^{-1}$)')
    else:
        print( 'z_name error' )
        raise

    plt.close()
        
    
    def y_slice_plot(control_index, control_gamma, control_V0):
        z = 1/hbar * np.sqrt( 2*me* (hv - 0 - E_work + control_V0) ) 
        ax.clear()
        mesh = ax.pcolormesh( z, x, data[:,control_index,:], cmap=cmp, zorder=0 )
        mesh.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[:,control_index,:]), vmax=np.max(data[:,control_index,:])) )
            
        ax.set_title( "Binding Energy = "+str(round(y[control_index],2))+' eV' )
        
        hv_cal = hbar**2 * np.arange( 0, BZ_len * 30, BZ_len )**2 / (2*me) + E_work - control_V0
        hv_cal = [str(round(x,2)) for x in hv_cal]
        new_labels = [ '\n\n'.join(x) for x in zip( x_ticks, hv_cal  ) ]
        ax.set_xticks(np.arange( 0, BZ_len * 30, BZ_len ), new_labels)
        ax.set_xlim([min(z)-0.5*BZ_len, max(z)+0.5*BZ_len])
        
        ax.grid(True)
    
        display(fig)
    

    interactive_plot = interactive(y_slice_plot, control_index=control_index, control_gamma=control_gamma, control_V0=control_V0)
    output = interactive_plot.children[-1]

    output.layout.height = '420px'

    ui1 = widgets.HBox( [control_V0, control_index, control_gamma, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_'+x_name+'_'+z_name+'_Eb_'+str(round(y[control_index.value],2))+'eV.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)    
    
    return fig, ax
    
    
    

    
    
def Affine_broadened_Fermi_edge_fit_plot(
        data, eb, hv, index_range, fermi_edge_center, fermi_edge_T, fermi_edge_conv_width, fermi_edge_const_bkg, fermi_edge_lin_bkg, fermi_edge_offset, sample_name, output_filepath
    ):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    
    def affine_broadened_fd(
            x, center=0, T=30, conv_width=0.02, const_bkg=1, lin_bkg=0, offset=0
        ):
        """Fermi function convoled with a Gaussian together with affine background.

        Args:
            x: value to evaluate function at
            fd_center: center of the step
            fd_width: width of the step
            conv_width: The convolution width
            const_bkg: constant background
            lin_bkg: linear background slope
            offset: constant background
        """
        dx = center-x
        x_scaling = np.abs(x[1] - x[0])
        fermi = 1 / (np.exp(dx / (8.617e-5 * T) ) + 1)
        return (
            scipy.ndimage.gaussian_filter((const_bkg + lin_bkg * dx) * fermi, sigma=conv_width / x_scaling)
            + offset
        )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [2, 1, 1]}, figsize=(1600*px, 350*px))
    
    x = eb[index_range[0]:index_range[1]]
    x_fit = np.linspace(x[0], x[-1], num=100, endpoint=True)

    control_hv_index = widgets.IntSlider( min=0, max=len(hv)-1, step=1, value=0, description='hv Index', continuous_update=True )
    
    if_axvline = widgets.Checkbox( value=True, description='Fermi Edge' )

    button = widgets.Button(description="Savefig")

    Eb_0_index = find_value_index(eb, 0)[0]
    curve_data = ax1.plot( x, data[index_range[0]:index_range[1],control_hv_index.value], zorder=0, label='Data', marker='+' )
    curve_fit = ax1.plot( x_fit,  affine_broadened_fd(
                                x_fit,
                                fermi_edge_center[control_hv_index.value], fermi_edge_T[control_hv_index.value], fermi_edge_conv_width[control_hv_index.value], 
                                fermi_edge_const_bkg[control_hv_index.value], fermi_edge_lin_bkg[control_hv_index.value], fermi_edge_offset[control_hv_index.value]
                                ), zorder=1, label='Fit')  
    ax1.set_xlabel("Binding Energy (eV)")
    #ax1.set_yticks([])
    ax1.set_xlim(np.min(x), np.max(x))
    vline = ax1.axvline(0, linestyle='--', color='orange')
    ax1.legend(loc=2)
    
    ax2.plot(hv, fermi_edge_center, zorder=0)
    scatter = ax2.scatter(hv[control_hv_index.value], fermi_edge_center[control_hv_index.value], color='r', zorder=1)
    ax2.set_xlabel("hv (eV)")
    ax2.set_ylabel("Offset (eV)")
    
    ax3.plot(hv, fermi_edge_T, zorder=0)
    scatter2 = ax3.scatter(hv[control_hv_index.value], fermi_edge_T[control_hv_index.value], color='r', zorder=1)
    ax3.set_xlabel("hv (eV)")
    ax3.set_ylabel(r"$T$ (K)")

    plt.close()
    
    def hv_slice_plot(control_index, if_axvline):
        curve_data[0].set_ydata(data[index_range[0]:index_range[1],control_index])
        curve_fit[0].set_ydata(affine_broadened_fd(
                                x_fit,
                                fermi_edge_center[control_index], fermi_edge_T[control_index], fermi_edge_conv_width[control_index], 
                                fermi_edge_const_bkg[control_index], fermi_edge_lin_bkg[control_index], fermi_edge_offset[control_index]
                                ))
        ax1.relim()
        ax1.autoscale(axis='y')
        ax1.set_title("hv = "+str(round(hv[control_index],2))+" eV")
        
        scatter.set_offsets([hv[control_index], fermi_edge_center[control_index]])
        ax2.set_title('Offset = '+str(round(fermi_edge_center[control_index],4))+ ' eV' )
        
        scatter2.set_offsets([hv[control_index], fermi_edge_T[control_index]])
        ax3.set_title(r'$T$ = '+str(round(fermi_edge_T[control_index],2))+ ' K' )
        
        if if_axvline:
            vline.set_alpha(1)
        else:
            vline.set_alpha(0)  
    
        display(fig)
    

    interactive_plot = interactive(hv_slice_plot, control_index=control_hv_index, if_axvline=if_axvline)
    output = interactive_plot.children[-1]

    output.layout.height = '360px'

    ui1 = widgets.HBox( [control_hv_index, if_axvline, button] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)

    def on_button_clicked(b):
        fig.savefig(output_filepath + sample_name[1]+'_Fermi_edge_fit_'+str(hv[control_hv_index.value])+'eV.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
    button.on_click(on_button_clicked)
    
    return fig, ax1, ax2, ax3




def kp_align(data, x, y, z, sample_name, output_filepath, x_name='Y', y_name='Eb', z_name='hv'):
    fig = plt.figure(figsize=(8, 4))

    control_index = widgets.IntSlider( min=0, max=len(z)-1, step=1, value=0, description=z_name + ' Index', continuous_update=True )
    control_gamma = widgets.FloatSlider( min=0.1, max=2, step=0.1, value=1, description=r'$\gamma$ ratio', continuous_update=True )
    if_axhline = widgets.Checkbox( value=True, description='Fermi Edge', layout=widgets.Layout(width='10%'), indent=False )
    
    control_bar_len = widgets.FloatSlider( min=0, max=max(x), step=0.1, value=5, description=r'Bar len', continuous_update=True )
    control_bar_center = widgets.FloatSlider( min=-5, max=5, step=0.1, value=0, description=r'Offset', continuous_update=True )
    
    control_Eb_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=len(y)-20, description=y_name + ' Index', continuous_update=True )

    button = widgets.Button(description="Savefig", layout=widgets.Layout(width='10%'), indent=False )

    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    mesh1 = ax1.pcolormesh( x, y, np.transpose(data[:,:,control_index.value]), cmap=cmp, zorder=0 )
    mesh2 = ax2.pcolormesh( z, x, data[:,control_Eb_index.value,:], cmap=cmp, zorder=0 )
    
    hline1 = ax1.axhline(y=y[control_Eb_index.value], linestyle='--', linewidth=1, zorder=1, color='r')
    bar = ax1.plot([control_bar_center.value-control_bar_len.value/2.0, control_bar_center.value + control_bar_len.value/2.0],[0, 0], color='r')
    
    center_line1 = ax1.axvline(control_bar_center.value, color='r', linestyle='--')
    center_line2 = ax2.axhline(control_bar_center.value, color='r', linestyle='--', zorder=1)
    
    hv_line = ax2.axvline(z[control_index.value], color='r', linestyle='--', zorder=1)
    
    ax1.set_ylabel('Binding Energy (eV)')
    
    if x_name == 'Y':
        ax1.set_xlabel('Y (Deg)')
    elif x_name == 'kp':
        ax1.set_xlabel(r'$k_\parallel$ ($\AA^{-1}$)')
    else:
        print( 'x_name error' )
        raise
    ax1.invert_yaxis()
    ax1.set_ylim(top=-0.15)

    plt.close()
    
    def z_slice_plot(control_index, control_gamma, if_axhline, control_bar_len, control_bar_center, control_Eb_index):
        mesh1.set_array( np.transpose(data[:,:,control_index]) )
        mesh1.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[:,:,control_index]), vmax=np.max(data[:,:,control_index])) )
        
        mesh2.set_array( data[:,control_Eb_index,:] )
        mesh2.set_norm( colors.PowerNorm(gamma=control_gamma, vmin=np.min(data[:,control_Eb_index,:]), vmax=np.max(data[:,control_Eb_index,:])) )
        
        bar[0].set_xdata([control_bar_center-control_bar_len/2.0, control_bar_center + control_bar_len/2.0])
        hline1.set_ydata(y[control_Eb_index])
        center_line1.set_xdata(control_bar_center)
        center_line2.set_ydata(control_bar_center)
        hv_line.set_xdata(z[control_index])
        if if_axhline:
            hline1.set_alpha(1)
            #bar.set_alpha(1)
        else:
            hline1.set_alpha(0)
            #bar.set_alpha(0)
        
        if z_name == 'hv':
            ax1.set_title( "hv = "+str(round(z[control_index],2)) + ' eV' )
        elif z_name == 'kz':
            ax1.set_title( "kz = "+str(round(z[control_index],2)) + r' $\AA^{-1}$' ) 
        else:
            print( 'z_name error' )
            raise
       
        display(fig)
    

    interactive_plot = interactive(z_slice_plot, control_index=control_index, control_gamma=control_gamma, 
                                   if_axhline=if_axhline, control_bar_len=control_bar_len, control_bar_center=control_bar_center, control_Eb_index=control_Eb_index)
    output = interactive_plot.children[-1]

    output.layout.height = '420px'

    ui1 = widgets.HBox( [control_index, control_gamma, if_axhline, button] )
    ui2 = widgets.HBox( [control_Eb_index, control_bar_len, control_bar_center] )
    ui = widgets.VBox( [ui1, ui2, output] );
    display(ui)

    def on_button_clicked(b):
        if z_name == 'hv':
            fig.savefig(output_filepath + sample_name[1]+'_'+x_name+'_Eb_hv_'+str(z[control_index.value])+'eV.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
        elif z_name == 'kz':
            fig.savefig(output_filepath + sample_name[1]+'_'+x_name+'_Eb_kz_'+str(round(z[control_index],2))+'A^-1.png', format='png', transparent=True,
                    pad_inches = 0, bbox_inches='tight', dpi=300)
        else:
            print( 'z_name error' )
            raise
    button.on_click(on_button_clicked)
    
    return fig, ax1, ax2



def curve_interactive(data, x, y):
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    
    fig, ax = plt.subplots(1, 1, figsize=(1000*px, 300*px))
    

    control_index = widgets.IntSlider( min=0, max=len(y)-1, step=1, value=0, description='y Index', continuous_update=True )

    curve_data = ax.plot( x, data[:, control_index.value], zorder=1, label='Data', marker='+' ) 
    ax.set_xlim(np.min(x), np.max(x))
    
    plt.close()
    
    def curve_plot(control_index):
        curve_data[0].set_ydata( data[:, control_index])
        ax.relim()
        ax.autoscale(axis='y')
        ax.set_title('y = ' + str(round(y[control_index],2)) )
        
        display(fig)
    

    interactive_plot = interactive(curve_plot, control_index=control_index)
    output = interactive_plot.children[-1]

    output.layout.height = '320px'

    ui1 = widgets.HBox( [control_index] )
    ui = widgets.VBox( [ui1, output] );
    display(ui)
    
    return fig, ax
    
    