"""
Author: Sunil Mathew
Date: 29 June 2024
Concepts Studio
Get images from the internet, provides tools to crop, resize, and save images
Uses LLM to get related concepts
"""
import os
import glob
import time
import threading
import shutil
import re
import traceback
import json
import cv2
import imageio
import base64
import logging
import pyqtgraph as pg
import pandas as pd
# from facenet_pytorch import MTCNN
from mtcnn import MTCNN
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt

import concurrent.futures
from llama_cpp import Llama
from llama_cpp_agent.providers import LlamaCppPythonProvider

from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from core.config import config
from core.utils import get_time_hrs_mins_secs_ms, get_elapsed_time
from tasks.task_images import TaskImages


class ConceptsStudio:

    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(ConceptsStudio, cls).__new__(cls)
        return cls.instance

    def __init__(self, c=None, concepts_view=None, trial_config='dynamic_scr'):
        self.c = c
        self.img_size = 160
        self.crop_win_size = 200
        self.block_params = config["trial_config"][trial_config]["block_params"]
        self.force_replace = False
        self.related_concepts = True
        self.download_imgs = True
        self.crop_imgs = True
        self.popular_concepts = ['obama', 'trump', 'biden', 'putin', 'willsmith', 'tomcruise', 
                                 'angelinajolie', 'scarlettjohansson', 'bradpitt', 'leonardodicaprio', 
                                 'jenniferlopez', 'jenniferaniston', 'meganfox', 'spiderman', 'batman',
                                 'superman', 'wonderwoman', 'hulk', 'thor', 'captainamerica', 'ironman'
                                 'einstein', 'eddiemurphy', 'jimcarrey', 'jackiechan', 'brucelee', 'jetli',
                                 'arnoldschwarzenegger', 'sylvesterstallone', 'clinteastwood', 'tomhanks'
                                 'christmas', 'halloween', 'thanksgiving', 'valentinesday', 'easter', 'newyear']

        self.concepts_all = []
        self.downloaded_concepts = []
        self.patient_concepts = []
        self.related_concepts_dict = {}
        self.init_llm()
        self.init_facenet()
        self.task_images = TaskImages()
        if concepts_view is not None:
            self.concepts_view = concepts_view
            self.init_concepts_layout()

        responses_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'assets', 'concepts', config['responses_file']))
        main_pics_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'assets', config['pics_main_folder']))    
        # pics_xlsx_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'assets', 'concepts', config['pics_xlsx_file']))

        # if os.path.exists(pics_xlsx_file_path):
        #     self.load_concepts_pics_info(pics_xlsx_file_path)
        if os.path.exists(main_pics_path):
            self.update_pics_info(main_pics_path)

        if os.path.exists(responses_file_path):
            self.load_responses(responses_file_path)

    def init_facenet(self):
        self.mtcnn = MTCNN()
        # self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def init_llm(self):
        # Create an instance of the Llama class and load the model 
        self.llama_model = Llama(r"F:\LLM\\models\Meta-Llama-3.1-8B-Instruct.gguf", n_batch=512, n_threads=16, n_gpu_layers=40, n_ctx=2048, verbose=False)
        # self.llama_model = Llama(r"F:\LLM\\models\gemma-2b-it.Q8_0.gguf", n_batch=512, n_threads=16, n_gpu_layers=40, n_ctx=2048, verbose=False)
        # self.llama_model = Llama(r"F:\LLM\\models\qwen2-7b-instruct-q4_k_m.gguf", n_batch=256, n_threads=16, n_gpu_layers=40, n_ctx=2048, verbose=False)
        # self.llama_model = Llama(r"F:\LLM\\models\Lllama-3-RedElixir-8B.Q4_K_M.gguf", n_batch=256, n_threads=16, n_gpu_layers=40, n_ctx=2048)
        # self.llama_model = Llama(r"F:\LLM\\models\Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf", n_batch=512, n_threads=16, n_gpu_layers=40, n_ctx=2048)
        # self.llama_model = Llama(r"F:\LLM\\models\Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf", n_batch=512, n_threads=16, n_gpu_layers=40, n_ctx=2048)
        # Create the provider by passing the Llama class instance to the LlamaCppPythonProvider class
        provider = LlamaCppPythonProvider(self.llama_model)
        # Pass the provider to the LlamaCppAgentClass and define the system prompt and predefined message formatter
        self.concepts_agent = LlamaCppAgent(provider,
                      system_prompt="You are an advanced AI, tasked to create JSON responses structured " + 
                                    "as a list of objects(min=10) related to sports teams, tv shows, movies " + 
                                    "et. ex: for the prompt packers, response should be {\"name\":\"Aaron Rodgers\", \"char\":\"\", \"description\":\"Quarterback\"}, " + 
                                    "for frasier, response should be {\"name\":\"Kelsey Grammer\", \"char\":\"Frasier Crane\", \"description\":\"Psychiatrist\"}",
                      predefined_messages_formatter_type=MessagesFormatterType.CHATML)
        
    def image_to_base64_data_uri(self, file_path=None, img=None):
        if file_path is None and img is not None:
            base64_data = base64.b64encode(img).decode('utf-8')
            return f"data:image/png;base64,{base64_data}"
        elif file_path is not None and img is None:
            with open(file_path, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
    
    def update_pics_info(self, main_pics_path):
        """
        Update pictures info from the main pictures folder

        Args:
        	main_pics_path: Path to the main pictures folder
        """
        
        self.parent_concepts = []
        self.concepts_all = []
        self.img_paths = []
        face_imgs = glob.glob(os.path.join(main_pics_path, 'Faces', '**', '*.*'), recursive=True)
        self.img_paths.extend(face_imgs)
        for img in face_imgs:
            img_name = os.path.basename(img).split('.')[0]
            self.parent_concepts.append(img_name.split('~')[0])
            self.concepts_all.append(img_name.split('~'))
        imgs = glob.glob(os.path.join(main_pics_path, 'Non faces', '**', '*.*'), recursive=True)
        self.img_paths.extend(imgs)
        self.img_paths = list(set(self.img_paths))
        for img in imgs:
            img_name = os.path.basename(img).split('.')[0]
            self.concepts_all.append(img_name.split('~'))

        self.concepts_all = [item.split('_')[0] for sublist in self.concepts_all for item in sublist]
        self.concepts_all = list(set(self.concepts_all))
        print(f"Concept count: {len(self.concepts_all)}")

    def load_concepts_pics_info(self, file_path):
        """
        Load pictures info from an Excel file

        Args:
        	file_path: Path to the Excel file
        """
        self.concepts_pics_info = pd.read_excel(file_path)
        self.concepts_all = self.concepts_pics_info['Name'].values
        # some of them may contain more than one concept separated by ~
        self.concepts_all = [x.split('~') for x in self.concepts_all]
        self.concepts_all = [item for sublist in self.concepts_all for item in sublist]
        self.concepts_all = list(set(self.concepts_all))
        print(f"Concept count: {len(self.concepts_all)}")
        print(self.concepts_pics_info.head())

    def init_concepts_layout(self):
        if not hasattr(self, 'concepts_layout'):
            self.concepts_widget = QtWidgets.QWidget()
            self.lyt_concepts = QtWidgets.QVBoxLayout()
            self.concepts_widget.setLayout(self.lyt_concepts)

            # Controls 
            self.concepts_params_layout = QtWidgets.QHBoxLayout()
            self.lyt_concepts.addLayout(self.concepts_params_layout)
            
            self.init_concepts_cmb()
            self.init_concepts_spb()
            self.init_concepts_chk()
            self.init_concepts_btn()

            # Related concepts tree view
            self.related_concepts_tree_view = QtWidgets.QTreeView()
            self.related_concepts_tree_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
            self.related_concepts_tree_view.customContextMenuRequested.connect(self.concepts_tree_view_context_menu)
            self.related_concepts_tree_view.clicked.connect(self.concepts_tree_view_clicked)

            self.glw_concepts = pg.GraphicsLayoutWidget()
            self.glw_concepts.setContentsMargins(0, 0, 0, 0)
            self.glw_concepts.ci.layout.setSpacing(0)
            # self.glw_concepts.ci.geometryChanged.connect(self.update_response_plots_size)
            # self.concepts_view.addWidget(self.concepts_layout)

            self.concepts_scroll = QtWidgets.QScrollArea()
            self.concepts_scroll.setWidgetResizable(True)
            self.concepts_scroll.setWidget(self.glw_concepts)
            self.concepts_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            self.concepts_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)            
            self.concepts_view.addWidget(self.concepts_widget)

            self.lyt_hsplitter_concept_imgs = QtWidgets.QHBoxLayout()
            self.hsplitter_concept_imgs = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            self.lyt_hsplitter_concept_imgs.addWidget(self.hsplitter_concept_imgs)
            self.lyt_concepts.addLayout(self.lyt_hsplitter_concept_imgs)
            self.hsplitter_concept_imgs.addWidget(self.related_concepts_tree_view)
            self.hsplitter_concept_imgs.addWidget(self.concepts_scroll)
            self.lyt_concepts.setStretch(0, 1)
            self.lyt_concepts.setStretch(1, 100)

    def init_concepts_cmb(self):
        for param in config['concepts_cmb_params']:
            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(QtWidgets.QLabel(param))
            cmb = QtWidgets.QComboBox(self.concepts_widget)
            cmb.setObjectName(f'cmb_{param}')
            cmb.addItems(config['concepts_cmb_params'][param])
            cmb.currentIndexChanged.connect(self.update_concepts)
            vLyt.addWidget(cmb)
            self.concepts_params_layout.addLayout(vLyt)

    def init_concepts_spb(self):
        for param in config['concepts_spb_params']:
            # Spinbox for concepts parameters, use the config values to figure out if doublespinbox or spinbox
            if isinstance(config['concepts_spb_params'][param][0], float):                       
                spb = QtWidgets.QDoubleSpinBox(self.concepts_widget)               
            else:
                spb = QtWidgets.QSpinBox(self.concepts_widget)

            spb.setObjectName(f'spb_{param}')
            spb.setRange(config['concepts_spb_params'][param][1], config['concepts_spb_params'][param][2])
            spb.setValue(config['concepts_spb_params'][param][0])
            spb.setSingleStep(config['concepts_spb_params'][param][3])
            spb.valueChanged.connect(self.update_concepts_spb)
            lbl = QtWidgets.QLabel(param)

            vLyt = QtWidgets.QVBoxLayout()
            vLyt.addWidget(lbl)
            vLyt.addWidget(spb)
            self.concepts_params_layout.addLayout(vLyt)

    def update_concepts_spb(self):
        """
        Updates the concepts parameters.
        """
        sender = self.concepts_widget.sender()
        param = sender.objectName().replace('spb_', '')
        value = sender.value()

        if value == None:
            return
        else:
            if param == 'crop_win_size':
                setattr(self, param, value)

    def init_concepts_chk(self):
        for param in config['concepts_chk_params']:
            chk = QtWidgets.QCheckBox(param)
            chk.setObjectName(f'chk_{param}')
            chk.setChecked(config['concepts_chk_params'][param] == 'True')
            chk.stateChanged.connect(self.update_concepts_chk)
            self.concepts_params_layout.addWidget(chk)

    def update_concepts_chk(self):
        """
        Updates the concepts parameters.
        """
        sender = self.concepts_widget.sender()
        param = sender.objectName().replace('chk_', '')
        value = sender.isChecked()

        if value == None:
            return
        else:
            setattr(self, param, value)

    def init_concepts_btn(self):
        """
        Initialize concept studio buttons
        """
        for param in config['concepts_btn_params']:
            btn = QtWidgets.QPushButton(param)
            btn.setObjectName(f'btn_{param}')
            btn.clicked.connect(getattr(self, param))
            self.concepts_params_layout.addWidget(btn)

    def process(self):
        if self.related_concepts:
            self.process_related_concepts_threaded()
        elif self.download_imgs:
            self.download_imgs_ext()
        elif self.crop_imgs:
            self.crop_selected()

    def download_imgs_ext(self):
        """
        Download images
        """
        download_count = self.concepts_widget.findChild(QtWidgets.QSpinBox, 'spb_download_count').value()
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        output_dir = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient))
        if os.path.exists(output_dir):
            # Show a warning message box
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText(f'Are you sure you want to download images for {curr_patient}?')
            msg.setInformativeText("This action will overwrite existing images.")
            msg.setWindowTitle("Download Images")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = msg.exec_()
            if ret == QtWidgets.QMessageBox.Cancel:
                return
        self.c.log.emit(f"Downloading images for {curr_patient}...")
        self.c.progress.emit(0, f'Downloading images for {curr_patient}...')
        start_time = time.time()
        completed = 0
        tot_concepts = len(self.related_concepts_dict)
        for concept, related_concepts in self.related_concepts_dict.items():
            if concept not in self.concepts_all:
                output_dir = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept))
                self.task_images.download(concept=concept, limit=download_count, 
                                          output_dir=output_dir, adult_filter_off=True,
                                          filter='square', force_replace=self.force_replace, verbose=True)
            else:
                print(f"[Info] {concept} already exists. Skipping download.")
                # Color the concept green
                self.color_tree_item(concept, 'g')
            for related_concept in related_concepts:
                try:
                    related_concept_name = related_concept['char'].replace(' ', '').lower()
                    if related_concept_name == "":
                        related_concept_name = related_concept['name'].replace(' ', '').lower()
                    if related_concept_name not in self.concepts_all:
                        output_dir = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept_name))
                        self.task_images.download(concept=concept, related_concept=related_concept['char'], name=related_concept['name'], limit=5, output_dir=output_dir, adult_filter_off=True,
                                        filter='square', force_replace=self.force_replace, verbose=True)
                    else:
                        print(f"[Info] {related_concept['name']} already exists. Skipping download.")
                        self.color_tree_item(related_concept['name'], 'g')
                except:
                    print(traceback.format_exc())
            completed += 1
            progress = int(completed / tot_concepts * 100)
            hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
            # Remaining time = elapsed time / completed tasks * remaining tasks
            remaining_time = (hrs * 3600 + mins * 60 + secs) / completed * (tot_concepts - completed)
            hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
            # print(f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
            self.c.progress.emit(int(progress), f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')

        hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
        self.c.progress.emit(100, f'Downloading images for {curr_patient} completed in {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')
        self.update_related_concepts()

    def color_tree_item(self, concept, color, check=False):
        """
        Color the concept tree item
        """
        concept_item = self.related_concepts_model.findItems(concept, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
        if concept_item:
            concept_item[0].setBackground(QtGui.QBrush(QtGui.QColor(color)))
            concept_item[0].setCheckState(QtCore.Qt.Checked if check else QtCore.Qt.Unchecked)
        # Update the tree view
        self.related_concepts_tree_view.setModel(self.related_concepts_model)
    
    def download_img(self):
        """
        called via context menu to re-download an image
        """
        to_download = []

        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText() 
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item.text().lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            return
        elif item.text() == curr_patient:
            self.download_imgs()
            return
        elif item.parent().text() == curr_patient:
            concept = item.text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept))
            to_download.append(os.path.basename(file_path))
            output_dir = file_path   
        elif item.parent().parent().text() == curr_patient:
            related_concept = item.text().replace(' ', '').lower()
            concept = item.parent().text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept))
            to_download.append(os.path.basename(os.path.dirname(file_path)))
            output_dir = os.path.dirname(file_path)
            output_dir = os.path.abspath(os.path.join(output_dir, '..'))  
            
        self.task_images.download_images(output_dir=output_dir, names=to_download, force_replace=self.force_replace, filter='square', verbose=True)
        self.task_images.download(concept=concept, related_concept=related_concept['char'], name=related_concept['name'], limit=5, output_dir=output_dir, adult_filter_off=True,
                                        filter='square', force_replace=self.force_replace, verbose=True)
    
    def rename_concept(self):
        """
        Rename concept
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText() 
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item.parent().text() == curr_patient:
            concept = item.text()
            new_name, ok = QtWidgets.QInputDialog.getText(self.concepts_widget, 'Rename Concept', 'Enter new name:', text=concept)
            if ok:
                file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept))
                new_file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, new_name))
                os.rename(file_path, new_file_path)
                item.setText(new_name)
        elif item.parent().parent().text() == curr_patient:
            related_concept = item.text().replace(' ', '').lower()
            concept = item.parent().text()
            new_name, ok = QtWidgets.QInputDialog.getText(self.concepts_widget, 'Rename Concept', 'Enter new name:', text=related_concept)
            if ok:
                file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept))
                new_file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, new_name))
                os.rename(file_path, new_file_path)
                item.setText(new_name)
    
    def rename_img(self):
        """
        Rename image
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText() 
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item.text().lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            related_concept = item.parent().text().replace(' ', '').lower()
            concept = item.parent().parent().text()
            img = item.text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept, img))
            new_name, ok = QtWidgets.QInputDialog.getText(self.concepts_widget, 'Rename Image', 'Enter new name:', text=img)
            if ok:
                new_file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept, new_name))
                os.rename(file_path, new_file_path)
                item.setText(new_name)

    def delete_img(self):
        """
        Delete image
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText() 
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item.text().lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            related_concept = item.parent().text().replace(' ', '').lower()
            concept = item.parent().parent().text()
            img = item.text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept, img))
        elif item.text() == curr_patient:
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient))
        elif item.parent().text() == curr_patient:
            concept = item.text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept))
        elif item.parent().parent().text() == curr_patient:
            related_concept = item.text().replace(' ', '').lower()
            concept = item.parent().text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept))


        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

        # remove item from the model
        self.related_concepts_model.removeRow(index.row(), parent=index.parent())

    def delete_all(self):
        """
        Delete all downloaded images for the current patient
        """
        # Delete everything inside the task images path
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        folder_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient))
        if os.path.exists(folder_path):
            # Show a warning message box
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setText(f'Are you sure you want to delete all images downloaded for {curr_patient}?')
            msg.setInformativeText("This action cannot be undone.")
            msg.setWindowTitle("Delete All Images")
            msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
            msg.setDefaultButton(QtWidgets.QMessageBox.Cancel)
            ret = msg.exec_()
            if ret == QtWidgets.QMessageBox.Cancel:
                return
            try:
                shutil.rmtree(folder_path)
                curr_patient_item = self.related_concepts_model.findItems(curr_patient, QtCore.Qt.MatchExactly | QtCore.Qt.MatchRecursive)
                if curr_patient_item:
                    # remove all the children
                    for i in range(curr_patient_item[0].rowCount()):
                        curr_patient_item[0].removeRow(i)
            except:
                print(f"[Error] Deleting {folder_path} failed.")
                print(traceback.format_exc())

    def update_pics_main(self):
        """
        Update pictures main
        """
        self.update_crops_ids()
        self.c.progress.emit(0, 'Updating custom pics, pics now...')
        self.c.log.emit('Updating custom pics, pics now...')

        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        curr_patient_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient))
        if not os.path.exists(curr_patient_path):
            os.makedirs(curr_patient_path)
        if os.path.exists(curr_patient_path):
            custom_pics_path = os.path.abspath(os.path.join(curr_patient_path, 'custom_pics'))
            pics_now_path = os.path.abspath(os.path.join(curr_patient_path, 'pics_now'))
            os.makedirs(custom_pics_path, exist_ok=True)
            os.makedirs(pics_now_path, exist_ok=True)

        custom_pics_count = 0
        pics_now_count = 0
        copied_paths = []
        crops_path = os.path.abspath(os.path.join(curr_patient_path, 'crops'))
        
        for concept in self.related_concepts_dict:
            if concept in self.concepts_all:
                # Copy the images already available in the Pics\Faces, Pics\Non faces to the custom pics folder
                for img_path in self.img_paths:
                    img_name = os.path.basename(img_path).split('.')[0]
                    if concept in img_name and img_name.endswith('_1'):
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(custom_pics_path, os.path.basename(img_path))))
                            copied_paths.append(img_path)
                            custom_pics_count += 1
                            if custom_pics_count >= self.block_params["NPICS"][0]:
                                break
                        except:
                            print(f"[Error] Could not copy {img_path} to {custom_pics_path}.")
                    elif concept in img_name:
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(pics_now_path, os.path.basename(img_path))))
                            pics_now_count += 1
                        except:
                            print(f"[Error] Could not copy {img_path} to {pics_now_path}.")
            progress_perc = int(custom_pics_count / self.block_params["NPICS"][0] * 100)
            self.c.progress.emit(progress_perc, f'Updating custom pics, pics now...')
        # Copy the cropped images to the custom pics folder
        for root, dirs, files in os.walk(crops_path):
            for img in files:
                img_name = img.split('.')[0]
                if img_name.endswith('_1'):
                    try:
                        shutil.copy(os.path.abspath(os.path.join(root, img)), os.path.abspath(os.path.join(custom_pics_path, img)))
                        copied_paths.append(os.path.abspath(os.path.join(root, img)))
                        custom_pics_count += 1
                        if custom_pics_count >= self.block_params["NPICS"][0]:
                            break
                    except:
                        print(f"[Error] Could not copy {img} to {custom_pics_path}.")
                else:
                    try:
                        shutil.copy(os.path.abspath(os.path.join(root, img)), os.path.abspath(os.path.join(pics_now_path, img)))
                        pics_now_count += 1
                    except:
                        print(f"[Error] Could not copy {img} to {pics_now_path}.")
            progress_perc = int(custom_pics_count / self.block_params["NPICS"][0] * 100)
            self.c.progress.emit(progress_perc, f'Updating custom pics, pics now...')

        unused_concepts = list(set(self.popular_concepts) - set(self.patient_concepts))
        used_concepts = []
        if custom_pics_count < self.block_params["NPICS"][0]:
            for concept in unused_concepts:
                for img_path in self.img_paths:
                    if img_path in copied_paths:
                        continue
                    img_name = os.path.basename(img_path).split('.')[0]
                    if concept in img_name and img_name.endswith('_1'):
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(custom_pics_path, os.path.basename(img_path))))
                            copied_paths.append(img_path)
                            custom_pics_count += 1
                            if custom_pics_count >= self.block_params["NPICS"][0]:
                                break
                        except:
                            print(f"[Error] Could not copy {img_path} to {custom_pics_path}.")
                    elif concept in img_name:
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(pics_now_path, os.path.basename(img_path))))
                            pics_now_count += 1
                        except:
                            print(f"[Error] Could not copy {img_path} to {pics_now_path}.")
                progress_perc = int(custom_pics_count / self.block_params["NPICS"][0] * 100)
                self.c.progress.emit(progress_perc, f'Updating custom pics, pics now...')
                if custom_pics_count >= self.block_params["NPICS"][0]:
                    break

        if custom_pics_count < self.block_params["NPICS"][0]:
            unused_concepts = list(set(self.concepts_all) - set(self.patient_concepts) - set(used_concepts))
            for concept in unused_concepts:
                for img_path in self.img_paths:
                    if img_path in copied_paths:
                        continue
                    img_name = os.path.basename(img_path).split('.')[0]
                    if concept in img_name and img_name.endswith('_1'):
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(custom_pics_path, os.path.basename(img_path))))
                            copied_paths.append(img_path)
                            custom_pics_count += 1
                            if custom_pics_count >= self.block_params["NPICS"][0]:
                                break
                        except:
                            print(f"[Error] Could not copy {img_path} to {custom_pics_path}.")
                    elif concept in img_name:
                        try:
                            shutil.copy(img_path, os.path.abspath(os.path.join(pics_now_path, os.path.basename(img_path))))
                            pics_now_count += 1
                        except:
                            print(f"[Error] Could not copy {img_path} to {pics_now_path}.")
                progress_perc = int(custom_pics_count / self.block_params["NPICS"][0] * 100)
                self.c.progress.emit(progress_perc, f'Updating custom pics, pics now...')
                if custom_pics_count >= self.block_params["NPICS"][0]:
                    break
        copied_count = len(os.listdir(custom_pics_path))
        if copied_count < self.block_params["NPICS"][0]:
            self.c.progress.emit(100, f'Could not find enough custom pics for {curr_patient}. {copied_count} custom pics.')
            self.c.log.emit(f'Could not find enough custom pics for {curr_patient}. {copied_count} images copied.')
        else:
            self.c.progress.emit(100, f'Updating custom pics, pics now completed. {copied_count} custom pics.')
            self.c.log.emit(f'Updating custom pics, pics now completed. {copied_count} images copied.')
    
    def update_crops_ids(self):
        """
        Rename images to have _1, _2, _3, .. 
        for example: img_1.jpg, img_3.jpg, img_5.jpg will be renamed to img_1.jpg, img_2.jpg, img_3.jpg
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        crops_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, 'crops'))
        pic_id = 0
        for root, dirs, files in os.walk(crops_path):
            while pic_id < len(files):
                img = files[pic_id]
                img_name = img.split('.')[0]
                img_name[-1] = str(pic_id+1)
                img_ext = img.split('.')[1]
                new_img_name = f'{img_name}_{pic_id}.{img_ext}'
                os.rename(os.path.abspath(os.path.join(root, img)), os.path.abspath(os.path.join(root, new_img_name)))
                pic_id += 1

    def update_concepts(self):
        """
        Get concepts from the responses file for the selected patient
        """
        cmb = self.concepts_widget.sender()

        if cmb.objectName() == 'cmb_patient':
            curr_patient = cmb.currentText()
            if hasattr(self, 'responses_df'):
                patient_response = self.responses_df[self.responses_df['Full name'].str.contains(curr_patient, case=False)]
                # Get every column after the first 15 columns
                tot_columns = len(patient_response.columns)
                patient_response = patient_response.iloc[:, 15:tot_columns-11]
                # get only columns that are of type string
                patient_response = patient_response.select_dtypes(include=['object'])
                # print(patient_response.head())

                # Get the unique concepts from the responses first row column values
                concepts = patient_response.iloc[0, :].values

                # Remove nan, empty and duplicate values
                concepts = [re.split(';|\n|,', x.strip()) for x in concepts if isinstance(x, str) and x != '']
                concepts = [item.strip().lower() for sublist in concepts for item in sublist if item != '']
                concepts = list(set(concepts))

                self.related_concepts_dict = {}
                for concept in concepts:
                    self.related_concepts_dict[concept] = {}

                self.patient_concepts = concepts

        self.update_related_concepts()

    def concepts_tree_view_context_menu(self, position):
        """
        Handle right click event on the concepts tree view
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        # Get the selected item
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)
        menu = QtWidgets.QMenu()
        if item.text().lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            menu.addAction("Rename", self.rename_img)
            menu.addAction("Delete", self.delete_img)
        elif item.parent().text() == curr_patient:
            menu.addAction("Get related concepts", lambda: self.get_related_concepts_ext(item.text()))
            menu.addAction("Download", self.download_img)
            menu.addAction("Rename", self.rename_concept)
            menu.addAction("Delete", self.delete_img)
        else:
            menu.addAction("Download", self.download_img)
            menu.addAction("Rename", self.rename_concept)
            menu.addAction("Delete", self.delete_img)
        menu.exec_(self.related_concepts_tree_view.viewport().mapToGlobal(position))

    def concepts_tree_view_clicked(self):
        """
        Handle click event on the concepts tree view
        """
        # Get the selected item
        index = self.related_concepts_tree_view.currentIndex()
        item = self.related_concepts_model.itemFromIndex(index)

        if item is None:
            return
        
        # Get the selected item text
        concept = item.text()

        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        if item.text().lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
            related_concept = item.parent().text().replace(' ', '').lower()
            concept = item.parent().parent().text()
            img = item.text()
            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept, img))
            if os.path.exists(file_path):
                self.show_image(file_path=file_path, item=item)
            else:
                print(f"[Error] {file_path} does not exist.")
    
    def show_image(self, file_path, item):
        """
        Show the image in the concepts view

        Args:
        	file_path: Path to the image
        """
        if hasattr(self, 'vb'):
            vb.clear()
            vb = None
        # Clear the layout
        self.glw_concepts.clear()

        self.file_path = file_path

        # Add the image to the layout
        img = pg.ImageItem()
        imgData = imageio.imread(file_path)
        img.setImage(imgData)

        lyt_img = self.glw_concepts.addLayout(row=0, col=0)

        vb = lyt_img.addViewBox(row=0, col=0, lockAspect=True, invertY=True)
        vb.addItem(img)
        vb.autoRange()

        if not hasattr(self, 'img_info'):
            self.img_info = QtWidgets.QLabel()
            self.lyt_concepts.addWidget(self.img_info)
        
        self.img_info.setText(f'{imgData.shape[1]} x {imgData.shape[0]}')

        # if imgData.shape[0] != imgData.shape[1] or imgData.shape[0] > 1000:
        if item.data() is not None:
            faces_dict = item.data()
            if isinstance(faces_dict, dict):
                face_images = []
                for face_idx, face in faces_dict.items():
                    x1, y1, x2, y2 = face['x1'], face['y1'], face['x2'], face['y2']
                    prob = face['prob']
                    text = pg.TextItem(text=f'{prob:.2f} WxH:{x2-x1:.0f}x{y2-y1:.0f}', color='b', anchor=(0, 0), fill=(255, 255, 255, 100))
                    vb.addItem(text)
                    text.setPos(x1, y1+10)

                    crop_win = pg.RectROI([x1, y1], [x2-x1, y2-y1], scaleSnap=True, removable=True,
                                          maxBounds=pg.QtCore.QRectF(0, 0, imgData.shape[1], imgData.shape[0]),   
                                          pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashDotDotLine))
                    vb.addItem(crop_win)
                    crop_win.addScaleHandle([1, 1], [0, 0], lockAspect=True)
                    crop_win.sigRegionChanged.connect(lambda: self.update_crop(crop_win, text, prob, face_idx, faces_dict, file_path))
                    crop_win.sigRemoveRequested.connect(lambda: self.remove_roi(crop_win, text, faces_dict, face_idx))

                    # crop_data = imgData[y1:y2, x1:x2]
                    # crop_data = cv2.resize(crop_data, (224, 224))
                    # crop_data_uri = self.image_to_base64_data_uri(img=crop_data)
                    # response = self.llama_model.create_chat_completion(
                    #     messages = [
                    #         {
                    #             "role": "user",
                    #             "content": [
                    #                 {"type" : "text", "text": "Who or what is this in one or two words?"},
                    #                 {"type": "image_url", "image_url": {"url": crop_data_uri} }

                    #             ]
                    #         }
                    #     ]   
                    # )
                    # print(response["choices"][0]["message"]['content'])
                    # response_txt = pg.TextItem(text=response["choices"][0]["message"]['content'], color='b', anchor=(0, 0), fill=(255, 255, 255, 150))
                    # response_txt.setPos(x1, y2+10)
                    # vb.addItem(response_txt)

        else:
            self.detect_face_data(item, imgData, file_path, b_online=True, vb=vb)

        # mouse move event
        # vb.scene().sigMouseMoved.connect(self.mouse_moved)

        # mouse click event
        # vb.scene().sigMouseClicked.connect(self.mouse_clicked)

    def detect_face_data_facenet_mtcnn(self, item, img_data, file_path, b_online=False, vb=None):
        """
        Detect faces in the image and set the face data for each check item
        """
        if len(img_data.shape) > 2 and img_data.shape[2] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        elif len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        boxes, probs = self.mtcnn.detect(img_data)
        faces_dict = {}
        face_idx = 0
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                b_square = False
                if prob < 0.84:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # expand the box by 10% and make it square
                box_size = max(x2 - x1, y2 - y1)
                if box_size + 0.6 * box_size < img_data.shape[0] and \
                box_size + 0.6 * box_size < img_data.shape[1]:
                    box_size = int(box_size + 0.6 * box_size)
                hw = int(box_size / 2)
                x1 = center_x - hw
                x2 = center_x + hw
                y1 = center_y - hw
                y2 = center_y + hw

                x1 = max(0, x1)
                x2 = min(img_data.shape[1], x2)
                if x1 == 0 and box_size < img_data.shape[1]:
                    x2 = box_size
                elif x1 > 0 and x2 == img_data.shape[1] and x2 - box_size > 0:
                    x1 = x2 - box_size
                y1 = max(0, y1)
                y2 = min(img_data.shape[0], y2)
                if y1 == 0 and box_size < img_data.shape[0]:
                    y2 = box_size
                elif y1 > 0 and y2 == img_data.shape[0] and y2 - box_size > 0:
                    y1 = y2 - box_size
                    
                if x2 - x1 < 160 or y2 - y1 < 160:
                    # colr = 'y'
                    continue
                if (x2 -x1) - (y2 - y1) == 0:
                    colr = 'g'
                    b_square = True
                    faces_dict[face_idx] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'prob': prob, 'valid': True}
                else:
                    # colr = 'r'
                    # print(f'Invalid crop window for {os.path.basename(file_path)}: {x2 - x1}, {y2 - y1}')
                    continue
                
                if b_online:
                    self.add_roi(vb, img_data, x1, y1, x2, y2, prob, face_idx, faces_dict, file_path)

                face_idx += 1

            item.setData(faces_dict)
        if len(faces_dict) == 0:
            faces_dict[face_idx] = {'x1': 0, 'y1': 0, 'x2': self.crop_win_size, 'y2': self.crop_win_size, 'prob': 0.0, 'valid': False}
            if b_online:
                self.add_roi(vb, img_data, 0, 0, self.crop_win_size, self.crop_win_size, 0.0, 0, faces_dict, file_path)
            item.setData(faces_dict)

    def detect_face_data(self, item, img_data, file_path, b_online=False, vb=None):
        """
        Detect faces in the image and set the face data for each check item
        """
        if len(img_data.shape) > 2 and img_data.shape[2] == 4:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)
        elif len(img_data.shape) == 2:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
        faces = self.mtcnn.detect_faces(img_data)
        faces_dict = {}
        face_idx = 0
        if faces is not None:
            for face in faces:
                b_square = False
                prob = face['confidence']
                box = face['box']
                if prob < 0.84:
                    continue
                x1, y1, w, h = box
                x2 = x1 + w
                y2 = y1 + h
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                # expand the box by 10% and make it square
                box_size = max(x2 - x1, y2 - y1)
                if box_size + 0.6 * box_size < img_data.shape[0] and \
                box_size + 0.6 * box_size < img_data.shape[1]:
                    box_size = int(box_size + 0.6 * box_size)
                hw = int(box_size / 2)
                x1 = center_x - hw
                x2 = center_x + hw
                y1 = center_y - hw
                y2 = center_y + hw

                x1 = max(0, x1)
                x2 = min(img_data.shape[1], x2)
                if x1 == 0 and box_size < img_data.shape[1]:
                    x2 = box_size
                elif x1 > 0 and x2 == img_data.shape[1] and x2 - box_size > 0:
                    x1 = x2 - box_size
                y1 = max(0, y1)
                y2 = min(img_data.shape[0], y2)
                if y1 == 0 and box_size < img_data.shape[0]:
                    y2 = box_size
                elif y1 > 0 and y2 == img_data.shape[0] and y2 - box_size > 0:
                    y1 = y2 - box_size
                    
                if x2 - x1 < 160 or y2 - y1 < 160:
                    # colr = 'y'
                    continue
                if (x2 -x1) - (y2 - y1) == 0:
                    colr = 'g'
                    b_square = True
                    faces_dict[face_idx] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'prob': prob, 'valid': True}
                else:
                    # colr = 'r'
                    # print(f'Invalid crop window for {os.path.basename(file_path)}: {x2 - x1}, {y2 - y1}')
                    continue
                
                if b_online:
                    self.add_roi(vb, img_data, x1, y1, x2, y2, prob, face_idx, faces_dict, file_path)

                face_idx += 1

            item.setData(faces_dict)
        if len(faces_dict) == 0:
            faces_dict[face_idx] = {'x1': 0, 'y1': 0, 'x2': self.crop_win_size, 'y2': self.crop_win_size, 'prob': 0.0, 'valid': False}
            if b_online:
                self.add_roi(vb, img_data, 0, 0, self.crop_win_size, self.crop_win_size, 0.0, 0, faces_dict, file_path)
            item.setData(faces_dict)

    def add_roi(self, vb, img_data, x1, y1, x2, y2, prob, face_idx, faces_dict, file_path):
        # Add probability text
        text = pg.TextItem(text=f'{prob:.2f} WxH:{x2-x1:.0f}x{y2-y1:.0f}', 
                           color='b', anchor=(0, 0), fill=(255, 255, 255, 150))
        vb.addItem(text)
        text.setPos(x1, y2+10)

        crop_win = pg.RectROI([x1, y1], [x2-x1, y2-y1], scaleSnap=True, removable=True,
                                maxBounds=pg.QtCore.QRectF(0, 0, img_data.shape[1], img_data.shape[0]),
                                pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashDotDotLine))
        crop_win.addScaleHandle([1, 1], [0, 0], lockAspect=True)
        vb.addItem(crop_win)
        crop_win.sigRegionChanged.connect(lambda: self.update_crop(crop_win, text, 
                                                                   prob, face_idx, 
                                                                   faces_dict, file_path))
        
        crop_win.sigRemoveRequested.connect(lambda: self.remove_roi(crop_win, text,
                                                                        faces_dict, face_idx))

    def remove_roi(self, crop_win, text, faces_dict, face_idx):
        vb = crop_win.getViewBox()
        if vb is not None:
            vb.removeItem(crop_win)

        vb = text.getViewBox()
        if vb is not None:
            vb.removeItem(text)

        faces_dict.pop(face_idx, None)

    def update_crop(self, crop_win, text, prob, face_idx, faces_dict, file_path):
        x1, y1 = crop_win.pos()
        x1, y1 = int(x1), int(y1)
        x2, y2 = crop_win.size()
        x2, y2 = int(x2), int(y2)
        x2 += x1
        y2 += y1
        if x2 - x1 < 160 or y2 - y1 < 160:
            colr = 'y'
        elif (x2 -x1) - (y2 - y1) < 0.1:
            colr = 'g'
            faces_dict[face_idx] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'prob': prob, 'valid': True}
        else:
            colr = 'r'
            print(f'Invalid crop window for {os.path.basename(file_path)}: {x2 - x1}, {y2 - y1}')
        crop_win.setPen(pg.mkPen(colr, width=1, style=QtCore.Qt.DashDotDotLine))
        text.setPos(x1, y2+10)
        text.setText(f'{prob:.2f} WxH:{x2-x1:.0f}x{y2-y1:.0f}')

    def crop_selected(self):
        """
        Crop the checked images in concept tree view
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        # Get root item of the tree and traverse the tree to get the checked items
        root_item = self.related_concepts_model.invisibleRootItem()
        if root_item.hasChildren():
            for i in range(root_item.rowCount()):
                patient_item = root_item.child(i)
                if patient_item.text() == curr_patient:
                    # Create a new branch for the crops
                    crops_item = QtGui.QStandardItem('crops')
                    patient_item.appendRow(crops_item)
                    if patient_item.hasChildren():
                        for j in range(patient_item.rowCount()):
                            concept_item = patient_item.child(j)
                            if concept_item.checkState() == QtCore.Qt.Checked or \
                               concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                concept = concept_item.text()
                                crops_concept_item = QtGui.QStandardItem(concept)
                                crops_item.appendRow(crops_concept_item)
                                if concept_item.hasChildren():
                                    for k in range(concept_item.rowCount()):
                                        related_concept_item = concept_item.child(k)
                                        if related_concept_item.checkState() == QtCore.Qt.Checked or \
                                           related_concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                            related_concept = related_concept_item.text().replace(' ', '').lower()
                                            crops_related_concept_item = QtGui.QStandardItem(related_concept)
                                            crops_concept_item.appendRow(crops_related_concept_item)
                                            if related_concept_item.hasChildren():
                                                for l in range(related_concept_item.rowCount()):
                                                    img_item = related_concept_item.child(l)
                                                    if img_item.checkState() == QtCore.Qt.Checked:
                                                        img = img_item.text()
                                                        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                                                            file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept, img))
                                                            crop_file_path = os.path.abspath(os.path.join(self.task_images.task_images_path, curr_patient, 'crops', concept, related_concept, img))
                                                            if os.path.exists(file_path):
                                                                try:
                                                                    img_data = imageio.imread(file_path)
                                                                    if img_item.data() is not None:
                                                                        faces_dict = img_item.data()
                                                                    else:
                                                                        self.detect_face_data(item=img_item, img_data=img_data, file_path=file_path, b_online=False)
                                                                        faces_dict = img_item.data()
                                                                    if isinstance(faces_dict, dict):
                                                                        for face_idx, face in faces_dict.items():
                                                                            if face['valid']:
                                                                                crop_file_name = self.crop_save_image(img_data=img_data, 
                                                                                                    x1=face['x1'], x2=face['x2'], 
                                                                                                    y1=face['y1'], y2=face['y2'], 
                                                                                                    file_path=crop_file_path, face_idx=face_idx)
                                                                                crop_img_item = QtGui.QStandardItem(crop_file_name)
                                                                                crops_related_concept_item.appendRow(crop_img_item)
                                                                except:
                                                                    print(f"[Error] Crop failed for {file_path}.")
                                                                    print(traceback.format_exc())  

    def get_selected_concepts(self):
        """
        Get the selected concepts from the tree view
        """
        selected_concepts = []
        root_item = self.related_concepts_model.invisibleRootItem()
        if root_item.hasChildren():
            for i in range(root_item.rowCount()):
                patient_item = root_item.child(i)
                if patient_item.checkState() == QtCore.Qt.Checked or \
                   patient_item.checkState() == QtCore.Qt.PartiallyChecked:
                    # selected_concepts.append(patient_item.text())
                    if patient_item.hasChildren():
                        for j in range(patient_item.rowCount()):
                            concept_item = patient_item.child(j)
                            if concept_item.checkState() == QtCore.Qt.Checked or \
                               concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                selected_concepts.append(concept_item.text())
                                # if concept_item.hasChildren():
                                #     for k in range(concept_item.rowCount()):
                                #         related_concept_item = concept_item.child(k)
                                #         if related_concept_item.checkState() == QtCore.Qt.Checked or \
                                #            related_concept_item.checkState() == QtCore.Qt.PartiallyChecked:
                                #             selected_concepts.append(related_concept_item.text())
                                #             if related_concept_item.hasChildren():
                                #                 for l in range(related_concept_item.rowCount()):
                                #                     img_item = related_concept_item.child(l)
                                #                     if img_item.checkState() == QtCore.Qt.Checked:
                                #                         selected_concepts.append(img_item.text())
        print(selected_concepts)
        return selected_concepts
        
    def mouse_clicked(self, evt):
        """
        Handle mouse click event

        Args:
        	pos: Position of the mouse
        """
        vb = evt.scene()
        if not hasattr(self, 'lbl_pixel_value'):
            self.lbl_pixel_value = QtWidgets.QLabel()
            self.lyt_concepts.addWidget(self.lbl_pixel_value)
        # Get the position of the mouse
        pos = evt.scenePos()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
        else:
            return

        # Get the pixel value at the mouse position
        img = vb.addedItems[0]
        imgData = img.image
        if imgData is not None:
            x = int(mouse_point.x())
            y = int(mouse_point.y())
            if x >= 0 and x < imgData.shape[1] and y >= 0 and y < imgData.shape[0]:
                # Draw square crop window with mouse cursor at the center (white dotted lines)
                crop_window_size = self.crop_win_size
                x1 = x - crop_window_size // 2
                x2 = x + crop_window_size // 2
                y1 = y - crop_window_size // 2
                y2 = y + crop_window_size // 2
                
                self.crop_window.setData([x1, x1, x2, x2, x1], 
                                         [y1, y2, y2, y1, y1], 
                                         pen=pg.mkPen('g', width=1, style=QtCore.Qt.DashDotDotLine))
                vb.addItem(self.crop_window)

                # Crop and save the image
                self.crop_save_image(imgData, x1, x2, y1, y2)
        
    def crop_save_image(self, img_data, x1, x2, y1, y2, file_path=None, face_idx=0):
        """
        Crop and save the image
        """
        if file_path is None and hasattr(self, 'file_path'):
            crop_file_path = os.path.splitext(self.file_path)[0] + '_crop.png'
        elif file_path is not None:
            # append crop to the file name
            if face_idx > 0:
                crop_file_path = os.path.splitext(file_path)[0] + f'_{face_idx}.png'
            else:
                crop_file_path = os.path.splitext(file_path)[0] + '.png'
            crop_file_name = os.path.basename(crop_file_path)
        else:
            print("[Error] No file path provided, cropping skipped.")
            return
        crop_dir = os.path.dirname(crop_file_path)
        os.makedirs(crop_dir, exist_ok=True)
        # Crop the image and save it
        # convert to pixel coordinates
        x1 = max(0, x1)
        x2 = min(img_data.shape[1], x2)
        y1 = max(0, y1)
        y2 = min(img_data.shape[0], y2)
        cropData = img_data[y1:y2, x1:x2]
        
        # Save it as 160x160
        cropData = cv2.resize(cropData, (self.img_size, self.img_size))
        imageio.imwrite(crop_file_path, cropData, format='png')

        return crop_file_name

    def mouse_moved(self, pos):
        """
        Handle mouse move event

        Args:
        	pos: Position of the mouse
        """
        vb = pos.scene()
        if not hasattr(self, 'lbl_pixel_value'):
            self.lbl_pixel_value = QtWidgets.QLabel()
            self.lyt_concepts.addWidget(self.lbl_pixel_value)
        # Get the position of the mouse
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
        else:
            return

        # Get the pixel value at the mouse position
        img = vb.addedItems[0]
        imgData = img.image
        if imgData is not None:
            x = int(mouse_point.x())
            y = int(mouse_point.y())
            if x >= 0 and x < imgData.shape[1] and y >= 0 and y < imgData.shape[0]:
                pixel_value = imgData[y, x]
                self.lbl_pixel_value.setText(f'Mouse: [{x},{y}] Pixel Value: {pixel_value}')

                # Draw square crop window with mouse cursor at the center (white dotted lines)
                crop_window_size = self.crop_win_size
                x1 = x - crop_window_size // 2
                x2 = x + crop_window_size // 2
                y1 = y - crop_window_size // 2
                y2 = y + crop_window_size // 2

                if x1 < 0 or x2 >= imgData.shape[1] or y1 < 0 or y2 >= imgData.shape[0]:
                    col = 'y'
                else:
                    col = 'b'
                
                self.crop_window.setData([x1, x1, x2, x2, x1], 
                                         [y1, y2, y2, y1, y1], 
                                         pen=pg.mkPen(col, width=1, style=QtCore.Qt.DashDotDotLine))
                vb.addItem(self.crop_window, ignoreBounds=True)

    def load_responses(self, file_path):
        """
        Load responses from a file

        Args:
        	file_path: Path to the file
        """
        self.responses_df = pd.read_excel(file_path)
        self.responses_df = self.responses_df.fillna('')
        # Use columns First name and Last name to create a new column Patient and then add it to cmb_patient
        self.responses_df['Full name'] = self.responses_df['First name'] + ' ' + self.responses_df['Last name']

        cmb_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient')
        # cmb_patient.clear()
        try:
            cmb_patient.currentIndexChanged.disconnect(self.update_concepts)
        except:
            pass
        patient_names = self.responses_df['Full name'].values.tolist()
        patient_names = [x for x in patient_names if isinstance(x, str)] # remove nan

        cmb_patient.addItems(patient_names)
        cmb_patient.currentIndexChanged.connect(self.update_concepts)

    def process_related_concepts_threaded(self):
        try:
            proc_thread = threading.Thread(target=self.get_related_concepts)
            proc_thread.start()
        except:
            print(traceback.format_exc())

    def get_related_concepts(self):
        """
        Get related concepts for patient concepts in parallel
        """
        b_multi_thread = False
        tot_concepts = len(self.patient_concepts)
        completed = 0
        start_time = time.time()

        try:
            new_concepts = []
            # new_concepts = list(set(self.patient_concepts) - set(self.concepts_all))
            new_concepts = self.get_selected_concepts()
            if b_multi_thread:
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    for concept in new_concepts:
                        futures = [executor.submit(self.related_concepts, concept)]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(e)
                        completed += 1
                        progress_percentage = completed / tot_concepts * 100

                        if completed > 0:
                            hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
                            # Remaining time = elapsed time / completed tasks * remaining tasks
                            remaining_time = (hrs * 3600 + mins * 60 + secs) / completed * (tot_concepts - completed)
                            hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
                            # print(f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                            self.c.progress.emit(int(progress_percentage), f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')
                        else:
                            self.c.progress.emit(int(progress_percentage), None)
            else:
                self.c.progress.emit(0, f'Getting related concepts for {tot_concepts} concepts...')
                self.c.log.emit(f'Getting related concepts for {tot_concepts} concepts...')
                for concept in new_concepts:
                    self.get_related_concepts_ext(concept)
                    completed += 1
                    progress_percentage = completed / tot_concepts * 100
                    self.c.progress.emit(int(progress_percentage), None)
                    if completed > 0:
                        hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
                        # Remaining time = elapsed time / completed tasks * remaining tasks
                        remaining_time = (hrs * 3600 + mins * 60 + secs) / completed * (tot_concepts - completed)
                        hrs, mins, secs, ms = get_time_hrs_mins_secs_ms(seconds=remaining_time)
                        # print(f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds, {ms:.0f} ms.')
                        self.c.progress.emit(int(progress_percentage), f'Estimated remaining time: {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')
                        
            
            hrs, mins, secs, ms = get_elapsed_time(start_time=start_time)
            self.c.progress.emit(100, f'related concepts for {completed} concepts gathered in {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')
            self.c.log.emit(f'related concepts for {completed} concepts gathered in {hrs:.0f} hours, {mins:.0f} minutes, {secs:.0f} seconds')

            self.c.related_concepts_done.emit()      
        except:
            print(traceback.format_exc())

    def get_related_concepts_ext(self, concept):
        """
        Get related concepts for a concept

        Args:
        	concept: Concept
        """
        try:
            txt = self.concepts_agent.get_chat_response(concept)
            # Compile regex pattern
            pattern = r'\[\s*\{.*?\}\s*\]'

            # Search for the first match in the text
            match = re.search(pattern, txt, re.DOTALL)

            if match:
                # Extract and print only the first match
                json_resp = self.fix_and_load_json(match.group(0))
                self.related_concepts_dict[concept] = json_resp
                self.c.related_concepts_single_done.emit()
            else:
                print("No match found")
        except:
            self.related_concepts_dict[concept] = {}
            self.c.log.emit(f'Error getting related concepts for {concept}')
            print(traceback.format_exc())
            print(match.group(0) if match else txt)

    def fix_and_load_json(self, input_str):
        logging.basicConfig(level=logging.DEBUG)

        # Initial attempt to load JSON without modifications
        try:
            return json.loads(input_str)
        except json.JSONDecodeError as initial_error:
            logging.debug(f"Initial JSON parsing failed: {initial_error.msg}")

        # Fix missing commas between objects or arrays
        fixed_str = re.sub(r'}\s*(?=\{)', '},', input_str)  # Between objects
        fixed_str = re.sub(r']\s*(?=\{)', '],{', fixed_str)  # Between array and object
        fixed_str = re.sub(r'}\s*(?=\[)', '},[', fixed_str)  # Between object and array
        fixed_str = re.sub(r']\s*(?=\[)', '],[', fixed_str)  # Between arrays

        # Attempt to load the JSON after applying fixes
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError as fixed_error:
            logging.error(f"Error loading JSON after attempting to fix: {fixed_error.msg}")
            # Re-raise the error with additional context
            raise json.JSONDecodeError(f"Error loading JSON after attempting to fix: {fixed_error.msg}", fixed_error.doc, fixed_error.pos) from fixed_error
    
    def post_process_related_concepts(self):
        if self.download_imgs:
            self.download_imgs_ext()   
        if self.crop_imgs:
            self.crop_selected()  

        self.update_related_concepts(b_llm=True)

    def update_related_concepts(self, b_llm=False):
        """
        Add related concepts to the related concepts tree view
        """
        curr_patient = self.concepts_widget.findChild(QtWidgets.QComboBox, 'cmb_patient').currentText()
        # Clear the related concepts tree view
        self.related_concepts_tree_view.setModel(None)

        self.related_concepts_model = QtGui.QStandardItemModel()
        self.related_concepts_tree_view.setModel(self.related_concepts_model)
        self.related_concepts_tree_view.setRootIsDecorated(False)
        self.related_concepts_tree_view.setAlternatingRowColors(True)
        self.related_concepts_tree_view.setHeaderHidden(True)
        self.related_concepts_model.itemChanged.connect(self.check_uncheck_items)

        # Create a root item to check/uncheck all
        root_item = QtGui.QStandardItem(curr_patient)
        root_item.setCheckable(True)
        root_item.setCheckState(Qt.Checked)
        self.related_concepts_model.appendRow(root_item)

        if b_llm:
            for concept in self.related_concepts_dict:
                item = QtGui.QStandardItem(concept)
                item.setCheckable(True)
                item.setCheckState(Qt.Checked)
                if concept.replace(" ", "") in self.concepts_all:
                    item.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                    item.setCheckState(Qt.Unchecked)
                for key in self.related_concepts_dict[concept]:
                    try:
                        if key['char'] == '':
                            child = QtGui.QStandardItem(key['name'])
                            folder_name = key['name'].replace(" ", "").lower()
                        else:
                            child = QtGui.QStandardItem(key['char'])
                            folder_name = key['char'].replace(" ", "").lower()
                        
                        child.setToolTip(key['description'])
                        child.setCheckable(True)
                        child.setCheckState(Qt.Checked)
                        item.appendRow(child)
                        
                        path = os.path.join(self.task_images.task_images_path, curr_patient, concept, folder_name)
                        if os.path.exists(path):
                            imgs = os.listdir(path)
                            for img in imgs:
                                img_item = QtGui.QStandardItem(img)
                                img_item.setCheckable(True)
                                img_item.setCheckState(Qt.Checked)
                                child.appendRow(img_item)
                    except:
                        print(f"[Error] {key} of {concept} not formatted correctly. Skipping.")

                root_item.appendRow(item)
        else:
            for concept in self.related_concepts_dict:
                item = QtGui.QStandardItem(concept)
                item.setCheckable(True)
                item.setCheckState(Qt.Checked)
                if concept.replace(' ', '') in self.concepts_all:
                    # Color the concept if it is in the concepts_all list
                    item.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                    item.setCheckState(Qt.Unchecked)
                path = os.path.join(self.task_images.task_images_path, curr_patient, concept)
                if os.path.exists(path):
                    related_concepts = os.listdir(path)
                    for related_concept in related_concepts:
                        child = QtGui.QStandardItem(related_concept)
                        child.setCheckable(True)
                        child.setCheckState(Qt.Checked)
                        item.appendRow(child)
                        if related_concept in self.concepts_all:
                            # Color the related concept if it is in the concepts_all list
                            child.setBackground(QtGui.QBrush(QtGui.QColor(0, 255, 0)))
                        # Get the images for the related concept
                        path = os.path.join(self.task_images.task_images_path, curr_patient, concept, related_concept)
                        if os.path.exists(path):
                            imgs = os.listdir(path)
                            for img in imgs:
                                img_item = QtGui.QStandardItem(img)
                                img_item.setCheckable(True)
                                img_item.setCheckState(Qt.Checked)
                                child.appendRow(img_item)

                root_item.appendRow(item)

        self.related_concepts_tree_view.expandAll()

    def check_parent(self, parent):
        if not parent:
            return
        checked_count = 0
        total_count = parent.rowCount()
        for row in range(total_count):
            if parent.child(row).checkState() != Qt.Unchecked:
                checked_count += 1
        if checked_count == 0:
            parent.setCheckState(Qt.Unchecked)
        elif checked_count == total_count:
            parent.setCheckState(Qt.Checked)
        else:
            parent.setCheckState(Qt.PartiallyChecked)
        self.check_parent(parent.parent())

    def check_uncheck_items(self, item, update_selection=True):
        self.related_concepts_model.itemChanged.disconnect(self.check_uncheck_items)
        
        if item.hasChildren():
            self.check_children(item, item.checkState())
        
        self.check_parent(item.parent())
        
        self.related_concepts_model.itemChanged.connect(self.check_uncheck_items)

    def check_children(self, parent, check_state):
        for row in range(parent.rowCount()):
            child = parent.child(row)
            child.setCheckState(check_state)
            if child.hasChildren():
                self.check_children(child, check_state)

    def update_exp_config(self, trial_config):
        self.block_params = config["trial_config"][trial_config]["block_params"]

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    concepts_view = QtWidgets.QVBoxLayout()
    concepts_studio = ConceptsStudio(concepts_view=concepts_view)
    concepts_studio.concepts_widget.show()
    app.exec_()