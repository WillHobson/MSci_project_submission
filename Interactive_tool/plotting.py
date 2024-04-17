import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML, Image
from ipywidgets import FileUpload
import os
import pandas as pd
import shrinkingcoremodel as scm
import warnings
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, push_notebook
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.layouts import column
from traitlets import HasTraits, Float, link
from ipywidgets import FloatSlider, interact

class Plots:
    def __init__(self, SM, BM, SB):
        #Use the shrinking core simulation class
        self.inst_sm = SM
        self.inst_bm = BM
        self.inst_sb = SB
        
        #create widgets to hold plots 1,2,3
        #self.plot2_output = widgets.Output(layout={'width': '500px', 'height': '405px'})
        #self.plot3_output = widgets.Output(layout={'width': '500px', 'height': '405px'})
        #self.plot2_output = widgets.Output()
        #self.plot3_output = widgets.Output()
        self.metadata = widgets.Output(layout={'width': '450px', 'height': '50px'})
        self.plot_output = widgets.Output(layout={'width': '420px', 'height': '505px'})
        self.plot4_output = widgets.Output(layout={'width': '420px', 'height': '110px'})
        self.plot5_output = widgets.Output(layout={'width': '900px', 'height': '50px' })
        self.plot6_output = widgets.Output()
        #self.plot10_output = widgets.Output()
        self.model_descriptions()
        
        self.plot41_output = widgets.Output(layout={'width': '420px', 'height': '35px'})
        
        self.plot_results = widgets.Output()

 

    def smallfig(self):
        with self.plot4_output:
            plt.figure(facecolor='#ADD8E6')  # Set figure size to match widget size
            plt.plot([0, 1], [1, 1], color='#ADD8E6')
            plt.axis('off')  # Turn off axis for a cleaner appearance
            plt.show()
            
    def model_descriptions(self):
        with self.plot5_output:
            
            text = 'The three models presented here are all derived from different principles and have different assumptions. The following links will provide a summary and mathematical derivation of each, providing the relevant literature: SCM, SB, CK'

            styled_text = (
                '<p style="font-family: Arial, Helvetica, sans-serif; font-size: 16px;text-align: justify;">'
                f'{text}'
                '</p>')

            # Display the HTML-formatted text
            display(HTML(styled_text))

            
            
        #function to update the plot showing simulation radius vs time        
#     def update_plot3(self, r,t,p, model):
#         self.plot3_output.clear_output(wait=True)
#         self.plot3_output.layout.height = '405px'
#         self.plot3_output.layout.width = '500px'

#         #self.plot3_output = widgets.Output(layout={'width': '500px', 'height': '405px'})
#         #clear previous display
        
#         #on the cleared display plot ..
#         output_notebook(hide_banner=True)
#         p3 = figure(title= f'CORE RADIUS vs TIME (R={r} T={t} P={p})',
#                     width = 400, height = 400, x_axis_label = "time (s)", 
#                     y_axis_label = 'Radius of Uranium core (um)')
#         p3.min_border_bottom = 120
#         p3.sizing_mode = 'scale_both'
        
#         with self.plot3_output:
#             #new SCM simulation using inputted values
#             sim = self.inst_sm.simulate(r,t,p)
#             x = sim[0]
#             y = sim[1]
#             y_outer = sim[3]
#             p3.line(x, y, legend_label = 'Core radius (SCM)')
#             p3.line(x, y_outer, legend_label = 'Particle radius (SCM)', line_color='red')
#             p3.legend.location = "bottom_left"
#             show(p3)  
          
            
            
#     def plot2(self, result, model, radius, pressure, temp):
#         if model == 1:
#             sim = result
#         elif model == 2:
#             print('model 2')
#             sim = result
#         else: 
#             sim = result
        
#         self.plot2_output.clear_output(wait=True)
#         self.plot2_output.layout.height = '805px'
#         self.plot2_output.layout.width = '500px'        

#         with self.plot2_output:
#             p2 = figure(title= f'PROGRESSION vs TIME (R={radius} T={temp} P={pressure})',width = 400, height = 400,x_axis_label = "time (s)",
#                         y_axis_label = "Reaction progression")
                    

#             p2.min_border_bottom = 120
#             p2.sizing_mode = 'scale_both'
#             x = sim[0]
#             y = (sim[2]-np.min(sim[2]))
#             y= ((y/np.max(y))-1)*-1
#             self.save_fitting_data(x,y)

#             r = p2.line(x, y, legend_label = 'SCM')
#             p2.legend.location = "bottom_right"
            
#             def update(T=temp):
#                 bn = self.scm(radius, T, pressure)
#                 ybn = bn[2]-np.min(bn[2])
#                 ybn = ((ybn/np.max(ybn))-1)*-1
#                 xbn = bn[0]
#                 r.data_source.data['y'] = ybn
#                 r.data_source.data['x'] = xbn
#                 push_notebook()
#             show(p2, notebook_handle=True)
#             interact(update,T=(273,973)) 
            
#     def plot10(self,result, model, radius, pressure, temp):
#         self.plot10_output.clear_output(wait=True)

#         self.plot10_output.layout.height = '575px'
#         self.plot10_output.layout.width = '500px'
#         sim=result
#         #find max time if temp is ma (973K)
#         mx = self.scm(radius, 973, pressure)[0][-1]
        
#         #use this to fix the x axis
       
#         with self.plot10_output:
#             x = sim[0]
#             y = (sim[2]-np.min(sim[2]))
#             y= ((y/np.max(y))-1)*-1
#             self.save_fitting_data(x,y)

#             p = figure(title= f'PROGRESSION vs TIME (R={radius} T={temp} P={pressure})',
#                     width = 400, height = 400,
#                     y_range=(0,1),
#                        x_range = (0,mx),

#                     x_axis_label = "time (s)",
#                     y_axis_label = "Reaction progression")
#             p.sizing_mode = 'scale_both'
#             #p.min_border_bottom = 120
#             r = p.line(x, y, legend_label = 'SCM')
            
#             def update(T=temp, Diffusivity=1):
#                 bn = self.scm(radius, T, pressure)
#                 ybn = bn[2]-np.min(bn[2])
#                 ybn = ((ybn/np.max(ybn))-1)*-1
#                 xbn = bn[0]
#                 r.data_source.data['y'] = ybn
#                 r.data_source.data['x'] = xbn
#                 push_notebook()
#             show(p, notebook_handle=True)
#             interact(update,T=(273,973), Diffusivity=(0,100))

    def scm(self, radius, temp, pressure):
        res = self.inst_sm.simulate(radius,temp,pressure)
        return res
    
    def bm(self, radius, temp, pressure):
        res = self.inst_bm.simulate(radius,temp,pressure, True)
        return res
    
    def sb(self, k,n):
        res = self.inst_sb.simulate(k,n)
        return res
    
            


            
    def update_plot(self, change, model=None, value=None, params = None):
        if type(change)==str and value==None:
            p1 = figure(title= f'Experimental data display')
            self.plot_output.clear_output(wait=True)

            with self.plot_output:
                x=[]
                y=[]
                p1.line(x,y)
                output_notebook(hide_banner=True)
                show(p1)

        else:
            if type(change)==str:
                selected_function = change
            else:
                selected_function = change['new']
            self.plot_output.clear_output(wait=True)
            data = pd.read_csv('stakebake_data/'+selected_function)
            self.calc_exp_times(data)
            output_notebook(hide_banner=True)
            self.p1 = figure(title= f'Plot of {selected_function}')#, width = 400, height = 400)
            self.p1.min_border_bottom = 40
            self.p1.sizing_mode = 'scale_both'
            self.p1.xaxis.axis_label = 'Time, s'
            self.p1.yaxis.axis_label = 'Reaction Progression'
            data.columns = ['x', ' y']
            with self.plot_output:
                x = data['x']
                y=np.array(data[' y'])
                self.p1.x_range.start = 0
                self.p1.x_range.end = np.max(np.array(x))
                if np.max(y) > 1:
                    y = y/np.max(y)
                if value==100:
                    nan_i = np.isnan(self.xfit)
                    if len(nan_i) > 0:
                        nan_indices_array = np.where(nan_i)[0]
                        for i in nan_indices_array:
                            self.xfit = np.delete(self.xfit, i)
                            self.yfit = np.delete(self.yfit, i)

                    self.source = ColumnDataSource(data={'x': self.xfit, 'y': self.yfit})

                    self.calc_fit_times(self.xfit, self.yfit)

                    self.line = self.p1.line(self.xfit, np.array(self.yfit), line_color='red', legend_label=f'{model} fit')
                    self.p1.x_range.end = np.max(np.array(self.xfit))


                self.p1.circle(x, y, legend_label = selected_function)
                self.p1.legend.location = "bottom_right"
                if params != None:
                    if model == 'Stakebake':
                        k = float(params[0])
                        kl = k - k/2
                        ku = k + k/2
                        n = float(params[1])
                        def update(K=k, order=n):
                            bn = self.sb(K,order)
                            ybn = bn[1]
                            #ybn = ((ybn/np.max(ybn))-1)*-1
                            xbn = bn[0]
                            self.line.data_source.data['y'] = ybn
                            self.line.data_source.data['x'] = xbn
                            self.p1.x_range.end = np.max(np.array(self.xfit))
                            push_notebook()
                            self.kslider = K
                            self.nslider = order
                        show(self.p1, notebook_handle=True)
                        interact(update,K=(kl,ku, 0.001), order=(0,0.99, 0.001))
                        
                        
                        
                        
                    elif model == 'Shrinking Core':
                        radius=int(params[0])
                        #ru = radius + radius/2
                        #print(ru)
                        #rl = radius-radius/2
                        #print(rl)
                        temp=int(params[1])
                        pressure=params[2]
                        def update(T=temp, R=radius):
                            bn = self.scm(R, T, pressure)
                            ybn = bn[2]-np.min(bn[2])
                            ybn = ((ybn/np.max(ybn))-1)*-1
                            xbn = bn[0]
                            #ybn = ((ybn/np.max(ybn))-1)*-1
                            xbn = bn[0]
                            self.line.data_source.data['y'] = ybn
                            self.line.data_source.data['x'] = xbn
                            self.p1.x_range.end = np.max(np.array(self.xfit))
                            push_notebook()
                            self.Tslider = T
                            self.Rslider = R

                        show(self.p1, notebook_handle=True)
                    

                        interact(update,T=(273,1000), R=(0.1, 1000, 0.1))
                        
                    else:    
                        radius=float(params[0])
                        ru = radius + radius/2
                        rl = radius-radius/2
                        temp=int(params[1])
                        pressure=params[2]
                        def update(T=temp, R=radius):
                            bn = self.bm(R, T, pressure)
                            ybn = bn[1]

                            xbn = bn[0]
                            self.line.data_source.data['y'] = ybn
                            self.line.data_source.data['x'] = xbn
                            self.p1.x_range.end = np.max(np.array(self.xfit))
                            push_notebook()
                            self.Tslider = T
                            self.Rslider = R
                        
                   
                        show(self.p1, notebook_handle=True)
                        interact(update,T=(temp-10,temp+10), R=(rl,ru, radius/10))
                else:
                    show(self.p1, notebook_handle=True)
                    #interact(update,T=(273,973), Diffusivity=(0,100))
                
                

                
                
    def slider_updates(self):
        self.line.data_source.data['x'] = np.array(self.xfit)
        self.line.data_source.data['y'] = np.array(self.yfit)
        push_notebook()
        #with self.plot_output:
            #show(self.p1)
        
        
        
        #print(self.p1.renderers)
        #self.p1.renderers.pop(0)
        #self.line = self.p1.line(self.xfit, np.array(self.yfit), line_color='green')
        #with self.plot_output:
            #show(self.p1)
#         with self.plot_output:
#             print('made it here')
#             self.p1.line(self.xfit, np.array(self.yfit), line_color='red')
#             print('made it here')
#             self.p1.x_range.end = np.max(np.array(self.xfit))
#             show(self.p1)
#             print('made it here')
        

                

    def find_times(self, per, xvals, yvals):

        if xvals[0]>xvals[1]:
            xvals = np.roll(xvals[::-1],0)
            yvals = np.roll(yvals[::-1],0)
            for i in range(len(xvals)):
                if xvals[i] < 0:
                    xvals[i] = 0
        
        if yvals[0]>yvals[1]:

            yvals = np.roll(yvals[::-1],0)
            

        newx = np.linspace(0, max(xvals), 10000)
        
        newy = np.interp(newx,xvals,yvals)
        differences = np.abs(newy - per)
        #differences = np.abs( - per)
        #print(differences[0])
        index_closest = np.argmin(differences)
        #print(round(newx[index_closest],3))
        return round(newx[index_closest],3)
    
    def calc_exp_times(self, data):
        data.columns = ['x', ' y']
        x=data['x']
        y=np.array(data[' y'])
        if np.max(y) > 1:
            y = y/np.max(y)
        
        self.t50 = self.find_times(0.5, x,y)
        self.t65 = self.find_times(0.65, x,y)
        self.t75 = self.find_times(0.75, x,y)
        self.t85 = self.find_times(0.85, x,y)
        self.t95 = self.find_times(0.95, x,y)
        
    def calc_fit_times(self, x, y):
        x=np.array(x)
        y=np.array(y)
        self.ft50 = self.find_times(0.5, x,y)
        self.ft65 = self.find_times(0.65, x,y)
        self.ft75 = self.find_times(0.75, x,y)
        self.ft85 = self.find_times(0.85, x,y)
        self.ft95 = self.find_times(0.95, x,y)
        
    

        

        

        
    def results(self):

        exps = [self.t50,self.t65,self.t75,self.t85,self.t95]
        fits = [self.ft50,self.ft65,self.ft75,self.ft85,self.ft95]
        errors = []
        for i in range(len(exps)):
            diff = round(abs(exps[i]-fits[i])/exps[i],5)

            errors.append(round(diff*100,3))
            

            
        table_data = [
    ['-', '50%', '65%','75%','85%','95%'],
    ['Experiment', self.t50,self.t65,self.t75,self.t85,self.t95],
    ['Your fit', self.ft50,self.ft65,self.ft75,self.ft85,self.ft95], 
    ['% error', errors[0],errors[1],errors[2],errors[3],errors[4]]
]

        # Create HTML table string
        table_html = '<table border="1" style="font-size: 16px;">'
        for row in table_data:
            table_html += '<tr>'
            for cell in row:
                table_html += '<td>{}</td>'.format(cell)
            table_html += '</tr>'
        table_html += '</table>'
        self.plot_results.clear_output(wait=True)
        with self.plot_results:
            display(HTML(table_html))
        
        
            
    def save_fitting_data(self, xfit, yfit):
        self.xfit = xfit
        self.yfit = yfit
        
#     def create_animation(self, r,t,p,model):
#         self.plot6_output.layout.height = '405px'
#         self.plot6_output.layout.width = '950px'
#         with self.plot6_output:
#             print('CREATING ANIMATION - please wait')
#         res = self.inst_sm.simulate(r,t,p)
#         TIME = res[0]
#         ORAD = res[3]
#         CRAD = res[1]
#         PRES = res[2]
#         title = f'Rad={r}_Temp={t}_Press={p}'
#         self.inst_sm.visualise(TIME, ORAD, CRAD, PRES, title);

        

#     def update_animation(self,title):
#         self.plot6_output.clear_output(wait=True)
#         with open(title+".gif", "rb") as file:
#             image = file.read()

#         play = widgets.Image(
#             value=image,
#             format='gif'
#         )

#         with self.plot6_output:
#             display(play)
            
            


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
