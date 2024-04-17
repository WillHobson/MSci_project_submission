import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from ipywidgets import FileUpload
import os
import pandas as pd
import shrinkingcoremodel as scm
import blochmintz as bmm
import warnings
from numba import jit
from bokeh.models import ColumnDataSource, CustomJS, Slider






class Main:
    def __init__(self):
        import warnings
        from widgets import ButtWidg
        from plotting import Plots
        from master import FunctionPlotter, ShrinkingCore, BlochMintz
        from bokeh.plotting import figure, show
        from bokeh.io import output_notebook
        from bokeh.models import ColumnDataSource, CustomJS, Slider
        

        warnings.filterwarnings('ignore')
        SM = ShrinkingCore()
        BM = BlochMintz()
        SB = StakeBake()
        PL=Plots(SM, BM, SB)
        bw=ButtWidg(SM, BM, PL, SB)
       
        

        # Create an instance of the class
        self.function_plotter = FunctionPlotter(bw, PL)

class FunctionPlotter:

    def __init__(self, BW, PL):
        #initialise the instance of the button widgets class
        self.inst_bw = BW
        
        #initialise an instance of the plotting class
        self.inst_pl = PL
        

        
        #initialise an instance of an experimental plot
        self.inst_pl.update_plot('no display')
        
        # create vertical box containing radiobuttons and simulate button
        simbox = widgets.VBox([self.inst_bw.radiobuttons, self.inst_bw.simulate])
        simbox.layout.margin = '38px -75px 0px 0px'
        #simbox.layout.height = '500px'
        
        #create vertical box containing the parameter input boxes
        parambox = widgets.VBox([self.inst_bw.output_textboxes])
        parambox.layout.margin = '0px 40px 0px 0px'
        
        #align the simulation and parameter boxes horizontally
        box1 = widgets.HBox([simbox,parambox])
        
        #Arrange top left corner
        topleft = widgets.VBox([widgets.HBox([self.inst_bw.upload, self.inst_bw.dropdown]),self.inst_pl.metadata,box1, self.inst_pl.plot41_output, self.inst_pl.plot4_output])
        
        butts_below= widgets.HBox([self.inst_bw.clear_exp, self.inst_bw.update_table])
        #arange top right corner
        topright = widgets.VBox([self.inst_pl.plot_output, butts_below, self.inst_bw.slider_output])
        
        #align top right and top left corner horizontally
        toprow = widgets.HBox([topleft, topright])
        
        # Create an HTML widget for the titles of the 2 main sections
        title_html = widgets.HTML('<h3 style="margin-bottom: -15px; font-family: cursive;font-size: 20px; font-weight: normal;">Model Results</h3>')
        title_html2 = widgets.HTML('<h3 style="margin-bottom: -15px; font-family: cursive;font-size: 20px; font-weight: normal;">Modelling Tool</h3>')
        
        #create a widget which is a line separating sections
        separator_html = widgets.HTML('<div style="border-top: 2px solid #3498db; margin: 10px 0;"></div>')
        
        #aligning the outputs from the simulation in the second row of the GUI
        secondrow =self.inst_pl.plot_results
        #widgets.HBox([self.inst_pl.plot10_output])#self.inst_pl.plot3_output])
        
        #Third row is for the fit to experiment button and sliders
        thirdrow = widgets.HBox([self.inst_bw.fit_output])
        
        nnn = widgets.VBox([thirdrow])
        
        #anim_butts = widgets.HBox([self.inst_bw.anim_butt_output, self.inst_bw.anim_close_output])
        #Line everything up vertically
        display_layout = widgets.VBox([self.inst_pl.plot5_output, title_html2, separator_html, toprow, title_html, separator_html, nnn, self.inst_pl.plot6_output, secondrow])#anim_butts])

        #set background colour for GUI
        custom_style = '<style>.widget-container { background-color: #ADD8E6; }</style>'
        display(HTML(custom_style))
        
        #Display everything
        display(display_layout)
        

        
class ShrinkingCore:
    def simulate(self,r,t,p, diff=None):
        R = int(r)
        T = int(t)
        P = int(p)
        N = 100
        rho_u = 79663.866
        rho_uh3 =  45435.68 #0.045/(100**3)
        b = 0.67
        velocity = 0.5
        rho_h = 0.081
        p = 100000 #1 bar
        a=1.11e-10
        initial_pellet_size = 360
        r = scm.simulation(T,R,P,N,1e-10,rho_h,rho_u,rho_uh3,velocity)
        times = r[0]/100000

        crad = r[1]
        press = r[2]
        outerrad = r[6]
        return times, crad, press, outerrad
   
    def visualise(self,T,O,C,P,t):
        scm.visualise(T,O,C,P,t);
        
class BlochMintz:
    def simulate(self,r,t,p,const):
        v=4.8e-4
        m=0.9
        p = int(p)
        t = int(t)
        r = float(r)/1000000

        if const == True:
            r = bmm.simulation_const_p(p, t, m, v,r, 0.1, 31840, 0)
        else:
            r = bmm.simulation(p, t, m, v,r, 0.1, 31840, 0)
            
        return r[0], r[1]
    
class StakeBake:
    def simulate(self,K, n):
        K=float(K)
        n=float(n)
        wm=12.5
        if n != 1:
            Mo =1
            #calclaute upper limit of t based on stakebake paper
            upper_t_lim = (Mo**(1-n))/(K*(1-n))
            #create array of times between 0 and upper lim in steps of 0.1
            t=np.linspace(0,upper_t_lim, 10001)
            term3 = K*(n-1)
            term4 = Mo**(1-n)
            term5 = term3*term4*t
            power = round(1/(1-n),10)
            
            wt = (1- (1+(term5))**power)*wm # calculate wm
        else:
            t = np.linspace(0,150, 1001)
            #t = t/100
            wt = wm*(1- np.exp(-K*t))
        return t, wt/12.5
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
    
    
    