
This repository will track the University of Bristol MSci project by  Will Hobson and Emily Wright**

Project Title: Modelling Uranium Hydride Formation

Date: September 2023 - April 2024

Project Supervisor: Dr Ross Springell


<br>
<br>


**DESCRIPTIONS OF FILES:**
- **Interactive_tool**: contains all data and code files needed for the interactive tool

  - **stakebake_data**: contains all data required (not just stakebake)

  - **application.ipynb**: jupyter notebook which renders the interactive tool

  - **master.py**: python script containing classes to implement each mathematical model

  - **plotting.py**: python script containing functions involved with plotting using Bokeh

  - **widgets.py**: python script containing functions involved with functionality of ipywidgets (e.g. buttons, sliders)

- **Code and Results**: contains folders for each of the authors containing the code and ata which was used to produce all output plots in Results section of final report

<br>

**INTRUCTIONS FOR SETUP OF INTERACTIVE TOOL:**
1) Download or git clone the 'Interactive_tool' folder
2) Open this folder in a Jupyter Environment
3) Open the 'application.ipynb' file and run all cells
4) Use tool

<br>

**INSTRUCTIONS FOR USE OF INTERACTIVE TOOL:**
1) Upload (upload button) or chose from catalouge (dropdown) the experimental data you wish to plot
   - This will display the data on the Bokeh plot on screen
3) Select which model you wish to use (radio buttons)
4) Input which model parameters you wish to use (input fields)
5) Click Simulate Button
   - This will display the model on the same Bokeh plot as your data along with the table showing percentages differences
6) (OPTIONAL) Use the sliders to alter the model parameters in live time on the plot
7) To start a new simulation click the 'clear plot' button
