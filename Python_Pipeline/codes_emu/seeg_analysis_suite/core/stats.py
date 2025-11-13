import os
import threading
import traceback
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
from qtpy.QtWidgets import QTableWidgetItem
import numpy as np

class Stats:
    def __init__(self, ui=True) -> None:
        self.df = None 
        self.has_ui = ui
        self.init_stat_viewer()

    def init_stat_viewer(self):
        if self.has_ui:
            # create a figure
            self.fig = plt.figure()
            self.fig.patch.set_facecolor('lightgrey')
            self.can = FigureCanvas(self.fig)
            self.fig.subplots_adjust(wspace=0.05, hspace=0.25,
                                        top=0.95, bottom=0.05, left=0.05, right=0.97)
        

    def show_stats_thread(self, df):
        thread = threading.Thread(target=self.show_stats, args=(df,))
        thread.start()

    def show_stats(self, df=None):
        pass
        
    def show_stats_dep(self, df=None):

        if df is not None:
            self.df = df
        
        num_rows, num_cols = self.df.shape

        # Create a histogram for each column
        cols = 6
        col_idx = 1
        for i in range(num_cols//cols):
            for j in range(cols):
                ax = self.fig.add_subplot(num_cols//cols, cols, col_idx)
                ax.hist(self.df.iloc[:, col_idx], bins=50)
                ax.set_title(self.df.columns[col_idx])
                col_idx += 1

    def read_folder_data(self):
        if self.animal_group == '':
            return
        
        info_file = os.path.join('Human observed', f'{self.animal_group}.xlsx')
        self.info_df = pd.read_excel(info_file, sheet_name='Data') # usecols='A:G'
        # self.display_df(self.info_df)


        self.plot_df = pd.read_excel(os.path.join('AI', f'summary_{self.animal_group}.xlsx'))
        self.update_plot_df()
        #filter = {'Group':'H', 'Animal #': ''}
        #self.plot_swipe_feature('Attention', filter=filter)

    def parse_values(self, col, filter=None):
        df = self.plot_df
        if filter['Animal #'] != '':
            animal_id = filter['Group']+ filter['Animal #']
            df = df[df['Animal ID'] == animal_id]
        if filter['Test Date'] != '':
            df = df[df['Date'] == filter['Test Date']]
            
        valid = df[col].dropna()  
        
        if not valid.empty:
            total  = valid.str.split(' ').str[-1].str.strip('()')
            values = valid.str.split(' ').str[0].str.strip('[]').str.split('|')
            return [values.str[i].astype(int).mean() for i in range(3)] + [total.str[0].astype(int).mean()]
        else: 
            return [0, 0, 0, 0] 

    def display_df(self, df=None):
        try:
            if df is None:
                df = self.info_df
            else:
                self.df = df

            # Clear table
            self.tblSummary.clear()
            # Display dataframe in pyqt table widget
            # Get the number of rows and columns
            num_rows, num_cols = df.shape

            # Set the table widget dimensions
            self.tblSummary.setRowCount(num_rows)
            self.tblSummary.setColumnCount(num_cols)

            # Set the table headers by converting non-string columns to string
            col_names = [str(col) for col in df.columns.values]
            self.tblSummary.setHorizontalHeaderLabels(col_names)

            # Populate the table widget with data
            for row in range(num_rows):
                for col in range(num_cols):
                    # Check if item is string or float
                    if df.iloc[row, col] is None:
                        str_item = ''
                    else:
                        str_item = f'{df.iloc[row, col]:.2f}' if isinstance(df.iloc[row, col], float) else f'{df.iloc[row, col]}'
                    item = QTableWidgetItem(str_item)
                    self.tblSummary.setItem(row, col, item)
            
            self.tblSummary.resizeColumnsToContents()
        except:
            print(traceback.format_exc())

    def filter_data(self, filter):
        for row in range(self.tblSummary.rowCount()):
            
            match = True
            
            for column in range(self.tblSummary.columnCount()):
            
                header = self.tblSummary.horizontalHeaderItem(column).text()
                item = self.tblSummary.item(row, column)
                
                if header in filter:
                    val = filter[header]
                    if val not in item.text():
                        match = False
                        break
                    
            self.tblSummary.setRowHidden(row, not match)

    def plot_stats(self, filter):
        self.plot_df = self.df.copy()
        for key in filter.keys():
            if filter[key] != '':
                # convert filter value to column dtype
                filter_val = self.plot_df[key].dtype.type(filter[key])
                self.plot_df = self.plot_df[self.plot_df[key] == filter_val]

        if self.plot_df.empty:
            print(f'No data to plot. Filter {filter}')
            return
        
        # Create a subplot for every row
        self.fig.clear()
        num_rows, num_cols = self.plot_df.shape
        self.fig.subplots(num_rows, 1, sharex=True)
        row_ctr = 0
        # plot the spike_times column for all neurons
        for i, row in self.plot_df.iterrows():
            ax = self.fig.get_axes()[row_ctr]
            row_ctr += 1
            for j, spike_times in enumerate(row['spike_times']):
                for neuron in spike_times:
                    # create y values for each neuron
                    y = np.ones(len(neuron)) * j
                    ax.scatter(neuron, y, marker='v', label=f'Neuron {j}')
                    ax.set_title(f"SessionID: {row['sessionID']}   Stimulus: {row['stimulus_name']}   Site: {row['recording_site']}   ChannelNo: {row['channel_number']}   Cluster: {row['cluster_number']}   Neuron count: {j}", fontweight='bold')

        self.can.draw()