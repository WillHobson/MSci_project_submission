import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from ipywidgets import FileUpload, interact
import os
import pandas as pd
import shrinkingcoremodel as scm
import warnings
from io import BytesIO
import time


class ButtWidg:
    def __init__(self, SM,BM,PL, SB):
        
        #initialise an instance of plotting and shrinking core class
        self.inst_sm = SM
        self.inst_bm = BM
        self.inst_pl = PL
        self.inst_sb = SB
        
        #initialise all widgets
        self.upload_button()
        self.dropdown_button()
        self.radiobuttons()
        self.seemore()
        self.simulate()
        self.slider_output = widgets.Output()

        
        self.fit_output = widgets.Output()
        self.anim_butt_output = widgets.Output()
        self.anim_close_output = widgets.Output()
        self.clear_experiment()
        self.update_table()

        

        

        # Create an Output widget to hold the textboxes
        self.output_textboxes = widgets.Output()

        
        self.radiobuttons.observe(self.display_textboxes, names='value')
        
        self.micro_symbol = "\u03BC"


    #This is the dropdown button for experimental data selection        
    def dropdown_button(self):
        # find the current working directory
        self.current_directory = os.getcwd()
        #define a button widget where the options are the data files
        self.dropdown = widgets.Dropdown(
            options=sorted(self.get_file_list(self.current_directory+'/stakebake_data')),
            value = 'zero.csv')
        # Observe the widgets to track changes
        self.dropdown.observe(self.inst_pl.update_plot, names='value')
        self.dropdown.observe(self.show_meta, names='value')
        #self.dropdown.observe(lambda change: self.inst_pl.update_plot(change, 0), names='value')
        
        
        
    def show_meta(self, filename):
        self.inst_pl.metadata.clear_output(wait=True)
        filename=self.dropdown.value
        text1 = f'Experimental conditions for {filename}:<br>'
        if filename == "griffiths550.csv":
            text = r"T=323K, $P_{init}$=1738mbar,    $mass_{U}$=1.8g, rig volume = 4.8e-4m$^3$ <hr>"
        elif filename == "griffiths700.csv":
            text = "T=323K, $P_{init}$=1738mbar,    $mass_{U}$=1.8g<hr>"
        elif filename == "stakebake50s.csv":
            text = "T=323K, $P_{const}$=266mbar<hr>"
        elif filename == "stakebake91s.csv":
            text = "T=364K,   $P_{const}$=266mbar<hr>"
        elif filename == "stakebake150s.csv":
            text = "T=423K,   $P_{const}$=266mbar<hr>"
        elif filename == "stakebake206s.csv":
            text = "T=479K,   $P_{const}$=266mbar<hr>"  
        elif filename == "stakebake250s.csv":
            text = "T=523K,   $P_{const}$=266mbar<hr>"
        with self.inst_pl.metadata:
            display(widgets.HTMLMath(value=text1+text))
    
    
    
    
    
    
    def update_dropdown(self, filename):
        #self.dropdown_button()
        file_list = sorted(self.get_file_list(self.current_directory+'/stakebake_data'))
        filtered_list = [x for x in file_list if x != '.DS_Store']
        self.dropdown.options = filtered_list
        self.dropdown.value = filename

    
    #This function is used to get the file names from data directory
    def get_file_list(self, folder_path):
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
        

    #This button displays every parameter for the shrinking core sim.
    def seemore(self):
        self.seemore = widgets.Button(description='See all parameters')
        # Set the event handler for the button click
        self.seemore.on_click(self.toggle_plot_text_boxes)

   # Function which displays more parameters if seemore clicked and gets back to the plot
    def toggle_plot_text_boxes(self, b):
        with self.inst_pl.plot4_output:
            clear_output(wait=True)
            if self.seemore.description == 'Hide all parameters':
                self.inst_pl.smallfig()
                self.seemore.description = 'See all parameters'
                self.display_textboxes()
            else:
                self.tb1 = widgets.Text(value = '79663.866', layout=widgets.Layout(width='100px'))
                rho ="\u03C1"
                desc1 = widgets.Label(f'{rho}U, '+ '$molm^{-3}$', 
                                      layout=widgets.Layout(width='150px'))
                
                self.tb2 = widgets.Text(value = '45435.68', layout=widgets.Layout(width='100px'))
                desc2 = widgets.Label(f'{rho}UH3, '+ '$molm^{-3}$', 
                                      layout=widgets.Layout(width='150px'))
                
                self.tb3 = widgets.Text(value = '0.67',layout=widgets.Layout(width='100px'))
                desc3 = widgets.Label('b', layout=widgets.Layout(width='150px'))
                
                self.tb4 = widgets.Text(value = '0.5', layout=widgets.Layout(width='100px'))
                desc4 = widgets.Label('Velocity $\mu m s^{-1}$', layout=widgets.Layout(width='150px'))
                
                self.tb5 = widgets.Text(value = '0.081', layout=widgets.Layout(width='100px'))
                desc5 = widgets.Label(f'{rho} H, '+'$molm^{-3}$', 
                                      layout=widgets.Layout(width='150px'))
                
                self.tb6 = widgets.Text(value = '1.11e-10',layout=widgets.Layout(width='100px'))
                desc6 = widgets.Label('a', layout=widgets.Layout(width='150px'))
                
                
                w1  = widgets.HBox([desc1, self.tb1])
                w2  = widgets.HBox([desc2, self.tb2])
                w3  = widgets.HBox([desc3, self.tb3])                
                w4  = widgets.HBox([desc4, self.tb4])                
                w5  = widgets.HBox([desc5, self.tb5])                
                w6  = widgets.HBox([desc6, self.tb6])      
                
                lhs = widgets.VBox([w1, w2,w3])
                rhs = widgets.VBox([w4,w5,w6])
                lhs.layout.margin_right = '150px'
                rhs.layout.margin_left = '150px'
                
                lay = widgets.HBox([lhs,rhs])
                display(lay)
                self.seemore.description = 'Hide all parameters'     

    #Button which runs simulation            
    def simulate(self):
        self.simulate = widgets.Button(description='Simulate')
        self.simulate.on_click(self.on_simulate_click)
    
    #Fetches the users values from textboxes and passes them to plot3 where sim is run
    def on_simulate_click(self,b):
        self.inst_pl.plot4_output.clear_output()
        try:
            model = self.radiobuttons.value
            if model == 'Shrinking Core':
                mod = 1
                r = self.textbox1.value
                t = self.textbox2.value
                p = self.textbox3.value
                params = (r,t,p)
                sim = self.scm(r,t,p)
                
            elif model == 'Stakebake':
                mod = 2
                k = self.textbox1.value
                n = self.textbox2.value
                params = (k,n)
                sim = self.sb(k,n)
                
            elif model == 'Bloch Mintz':
                mod = 4
                r = self.textbox1.value
                t = self.textbox2.value
                p = self.textbox3.value
                params = (r,t,p)
                #is constant pressure checkbox checked?
                if self.constPcheckbox.value==True:
                    #do const pressure Bm calc
                    sim = self.bm(r,t,p,True)
                else:
                    #do variable pressure bm calc
                    sim = self.bm(r,t,p,False)

         
            #self.inst_pl.update_plot3(r,t,p,model)
            #self.inst_pl.plot10(sim, mod, r,p,t)
            self.inst_pl.plot_results.layout = {'width': '900px', 'height': '150px', 'border': '1px solid black'}
            self.inst_pl.save_fitting_data(sim[0],sim[1])
            self.inst_pl.update_plot(self.dropdown.value, self.radiobuttons.value, 100, params)
            if mod == 1:
                print('yes')
                self.inst_pl.plot_results.clear_output(wait=True)
                with self.inst_pl.plot_results.layout:
                    print('sorry results not working for Shrinking Core, please try another model')
            else:
                
                self.inst_pl.results()
            #self.de_slider()
            
#             self.fit()
#             self.show_anim()
#             self.close_anim()
            
        except:
            with self.inst_pl.plot4_output:
                self.inst_pl.plot4_output.clear_output()
                print('please select a model')
    
    
            
    
    def scm(self, radius, temp, pressure):
        res = self.inst_sm.simulate(radius,temp,pressure)
        return res
    
    def bm(self, radius, temp, pressure, const):
        res = self.inst_bm.simulate(radius, temp, pressure, const)
        
        return res
    
    def sb(self, k,n):
        res = self.inst_sb.simulate(k, n)
        return res
    
    
    
    #This allows for a selection between simulatory models
    def radiobuttons(self):
        self.radiobuttons = widgets.RadioButtons(
            options=['Shrinking Core', 'Stakebake', 'Bloch Mintz'],
            value=None,  # No default selection
            description='Select Model:',
            disabled=False)
    
    #Button to upload own data - needs work
    def upload_button(self):
        self.upload = FileUpload(accept='.csv', multiple=True)
        self.upload.observe(self.on_upload, names='_counter')
        
        
   # Handle CSV file upload event
    def on_upload(self,change):
        target_directory = self.current_directory + '/stakebake_data/'
        for filename, file_info in self.upload.value.items():
            content = file_info['content']
            target_path = os.path.join(target_directory, filename)

            # Check if the file is a CSV file
            if filename.endswith('.csv'):
                # Read thse CSV content and save to target directory
                df = pd.read_csv(BytesIO(content))
                df.to_csv(target_path, index=False)
                
        self.update_dropdown(filename)
    

    #Dispplays textboxes where simulation parameters can be inserted      
    def display_textboxes(self, change):
        self.output_textboxes.clear_output(wait=True)
        if self.radiobuttons.value == 'Stakebake':
            self.textbox1 = widgets.Text(value='0.1', description='',layout=widgets.Layout(width='60px'))
            desc1 = widgets.Label(r'Rate constant $s^{-1}$', layout=widgets.Layout(width='150px'))
            self.textbox2 = widgets.Text(value='0.9', description='',layout=widgets.Layout(width='60px'))
            desc2 = widgets.Label(r'Reaction order (n)', layout=widgets.Layout(width='150px'))
            with self.output_textboxes:
                display(widgets.HBox([desc1, self.textbox1]))
                display(widgets.HBox([desc2, self.textbox2]))
                
        elif self.radiobuttons.value == 'Bloch Mintz':
            self.textbox1 = widgets.Text(value='3', description='',layout=widgets.Layout(width='60px'))
            self.textbox1.observe(self.check_rad_value, names='value')
            self.textbox2 = widgets.Text(value='293', description='',layout=widgets.Layout(width='60px'))
            self.textbox2.observe(self.check_temp_value, names='value')
            self.textbox3 = widgets.Text(value='26600', description='',layout=widgets.Layout(width='60px'))
            self.textbox3.observe(self.check_pressure_value, names='value')
             # Create descriptions with longer text
            desc1 = widgets.Label(f'Initial radius ({self.micro_symbol}m)', layout=widgets.Layout(width='150px'))
            desc2 = widgets.Label('Temperature (K)', layout=widgets.Layout(width='150px'))
            desc3 = widgets.Label('Pressure (Pa)', layout=widgets.Layout(width='150px'))
            self.textbox4 = widgets.Text(value='0.00048',
                                         description='',
                                         layout=widgets.Layout(width='60px'))
            self.textbox4.observe(self.check_pressure_value, names='value')
            desc4 = widgets.Label(r'Gas Rig Volume (m$^3$)', layout=widgets.Layout(width='150px'))

            self.textbox5 = widgets.Text(value='0.9', description='',layout=widgets.Layout(width='60px'))
            self.textbox5.observe(self.check_pressure_value, names='value')
            desc5 = widgets.Label(r'mass of uranium (g)', layout=widgets.Layout(width='150px'))


            self.constPcheckbox = widgets.Checkbox(value = False)
            self.description_html = widgets.Label('Hold pressure constant?', layout=widgets.Layout(width='200px'))
            self.inst_pl.plot41_output.clear_output(wait=True)
            
            with self.output_textboxes:
                display(widgets.HBox([desc1, self.textbox1]))
                display(widgets.HBox([desc2, self.textbox2]))
                display(widgets.HBox([desc3, self.textbox3]))
                if self.radiobuttons.value == 'Bloch Mintz':
                    display(widgets.HBox([desc4, self.textbox4]))
                    display(widgets.HBox([desc5, self.textbox5]))
           
            with self.inst_pl.plot41_output:
                    display(widgets.HBox([self.description_html, self.constPcheckbox]))


        elif self.radiobuttons.value == 'Shrinking Core':  
            # Create multiple textboxes
            self.textbox1 = widgets.Text(value='300', description='',layout=widgets.Layout(width='60px'))
            self.textbox1.observe(self.check_rad_value, names='value')
            self.textbox2 = widgets.Text(value='293', description='',layout=widgets.Layout(width='60px'))
            self.textbox2.observe(self.check_temp_value, names='value')
            self.textbox3 = widgets.Text(value='26600', description='',layout=widgets.Layout(width='60px'))
            self.textbox3.observe(self.check_pressure_value, names='value')
             # Create descriptions with longer text
            desc1 = widgets.Label(f'Initial radius ({self.micro_symbol}m)', layout=widgets.Layout(width='150px'))
            desc2 = widgets.Label('Temperature (K)', layout=widgets.Layout(width='150px'))
            desc3 = widgets.Label('Pressure (Pa)', layout=widgets.Layout(width='150px'))
            


                    
            
            butt = self.seemore

            with self.output_textboxes:
                display(widgets.HBox([desc1, self.textbox1]))
                display(widgets.HBox([desc2, self.textbox2]))
                display(widgets.HBox([desc3, self.textbox3]))

                display(widgets.HBox([butt]))

        else:
            # Uncheck the checkbox and clear the output when it's unchecked
            self.checkbox.value = False
            self.close_textboxes(None)
            
    def check_rad_value(self, change):
        try:
            if float(change.new)<0 or float(change.new)>1000:
                with self.inst_pl.plot4_output:
                    print(f"Check radius is between 0 and 1000 {self.micro_symbol}m")
            else:
                with self.inst_pl.plot4_output:
                    self.inst_pl.plot4_output.clear_output()
                    
        except ValueError:
                with self.inst_pl.plot4_output:
                    self.inst_pl.plot4_output.clear_output()
                    print("Enter only numeric radii")
                    
    def check_temp_value(self, change):
        try:
            if float(change.new)<0 or float(change.new)>1000:
                with self.inst_pl.plot4_output:
                    print("Check temperature is between 0 and 1000 Kelvin")
            else:
                with self.inst_pl.plot4_output:
                    self.inst_pl.plot4_output.clear_output()
                    
        except ValueError:
                with self.inst_pl.plot4_output:
                    self.inst_pl.plot4_output.clear_output()
                    print("Enter only numeric temperatures")
                    
    def check_pressure_value(self, change):    
        try:
            if float(change.new)<0 or float(change.new)>1000000:
                with self.inst_pl.plot4_output:
                    print("please use reasonable pressure value in Pa")
            else:
                with self.inst_pl.plot4_output:
                    self.inst_pl.plot4_output.clear_output()
                    
        except ValueError:
                with self.inst_pl.plot4_output:
                    self.inst_pl.plot4_output.clear_output()
                    print("please enter only numeric pressure values")

            
#     def fit(self):
#         self.fit_output.clear_output(wait=True)
#         self.fit_button = widgets.Button(description='fit to experimental plot') 
#         self.fit_button.on_click(self.on_fitting_click)
#         with self.fit_output:            
#             display(self.fit_button)
            
#     def on_fitting_click(self, b):
#         self.inst_pl.update_plot(self.dropdown.value, self.radiobuttons.value, 100)
        
#     def show_anim(self):
#         self.anim_butt_output.clear_output(wait=True)
#         self.anim_button = widgets.Button(description='Show Animation')
#         self.anim_button.on_click(self.on_anim_click)
#         with self.anim_butt_output:
#             display(self.anim_button)
            
            
#     def check_for_anim_file(self, title):
#         anim_loc = self.current_directory+'/'+title+'.gif'  
#         if os.path.exists(anim_loc):
#             return 0
#         else:
#             print('waiting for file')
#             time.sleep(5)
#             with self.inst_pl.plot6_output:
#                 print('CREATING ANIMATION')
#             self.check_for_anim_file(title)
        
            
#     def on_anim_click(self,b):
#         r = self.textbox1.value
#         t = self.textbox2.value
#         p = self.textbox3.value
#         model = self.radiobuttons.value
#         title = f'Rad={r}_Temp={t}_Press={p}'
#         self.inst_pl.create_animation(r,t,p, model)
#         self.check_for_anim_file(title)
#         self.inst_pl.update_animation(title)
#         #sim_anim = self.inst_sm.simulate(360, 364, 26600)
        
#     def close_anim(self):
#         self.anim_close_output.clear_output(wait=True)
#         self.anim_close = widgets.Button(description='Close Animation')
#         self.anim_close.on_click(self.on_animclose_click)
#         with self.anim_close_output:
#             display(self.anim_close)
           
        
#     def on_animclose_click(self, b):
#         self.inst_pl.plot6_output.clear_output(wait=True)
#         self.inst_pl.plot6_output.layout.height = '0px'
#         self.inst_pl.plot6_output.layout.width = '0px'

    def clear_experiment(self):
        self.clear_exp = widgets.Button(description='Clear plot')
        self.clear_exp.on_click(self.on_clear_click)
        
    def update_table(self):
        self.update_table= widgets.Button(description='update table')
        self.update_table.on_click(self.redotable)
        
    def redotable(self, b):
        model = self.radiobuttons.value
        print(model)
        if model == 'Shrinking Core':
            mod = 1
            r = self.inst_pl.Rslider
            t = self.inst_pl.Tslider
            p = self.textbox3.value
            params = (r,t,p)
            sim = self.scm(r,t,p)

        elif model == 'Stakebake':
            mod = 2
            k = self.inst_pl.kslider
            n = self.inst_pl.nslider
            params = (k,n)
            sim = self.sb(k,n)

        elif model == 'Bloch Mintz':
            mod = 4
            r = self.inst_pl.Rslider
            t = self.inst_pl.Tslider
            p = self.textbox3.value
            params = (r,t,p)
            #is constant pressure checkbox checked?
            if self.constPcheckbox.value==True:
                #do const pressure Bm calc
                sim = self.bm(r,t,p,True)
            else:
                #do variable pressure bm calc
                sim = self.bm(r,t,p,False)

        self.inst_pl.save_fitting_data(sim[0],sim[1])
        self.inst_pl.calc_fit_times(sim[0],sim[1])
        self.inst_pl.results()
        
        
    def on_clear_click(self, b):
        self.inst_pl.update_plot('no display')
        
        
        
        
#     def de_slider(self):
#         self.slider_output.clear_output(wait=True)
#         with self.slider_output:
#             val = self.textbox1.value
#             valu = float(val)+ (float(val)/2)
#             vall = float(val)- (float(val)/2)
#             self.de_slider = widgets.FloatSlider(
                
#                 value=val,   # Initial value
#                 min=vall,    # Minimum value
#                 max=valu,     # Maximum value
#                 step=0.1,     # Step size
#                 description='Initial radius:',  # Label for the slider
#                 orientation='horizontal')  # Orientation of the slider (horizontal or vertical)
            
#             display(self.de_slider)
            
#         self.de_slider.observe(self.slider_change, names='value')
        
#     def slider_change(self, b):
#         r = b['new']
#         t = self.textbox2.value
#         p = self.textbox3.value
#         if self.constPcheckbox.value==True:
#             sim = self.bm(r,t,p,True)
#         else:
#             sim = self.bm(r,t,p,False)
#         self.inst_pl.save_fitting_data(sim[0],sim[1])
#         self.inst_pl.slider_updates()   
        

        #self.inst_pl.update_plot(self.dropdown.value, self.radiobuttons.value, 100)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

                   

        