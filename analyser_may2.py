import sys
import os
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QVBoxLayout, QWidget, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
import numpy as np
from PyQt5.uic import loadUi
from python_speech_features import mfcc
import time
import soundfile as sf
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, Qt, QDateTime, QTimer
from PyQt5.QtGui import QIcon
from scipy.fft import fft, fftfreq
import scipy.signal as sg
from aud_edit02 import EditorWindow
import matplotlib
from scipy.signal import spectrogram
from PyQt5.QtGui import QDesktopServices
from matplotlib.ticker import MultipleLocator
from scipy.fftpack import dct
from PyQt5 import QtCore



class VoiceAnalyzer(QMainWindow):
    default_canvas=(1431/100,761/100)


    def __init__(self):
        super().__init__()

        loadUi('mainwindow.ui', self)
        
        self.dtDateTime.setDateTime(QDateTime.currentDateTime())
        
        self.setWindowTitle("Voice Analyser")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)

        self.figure = plt.figure(figsize=(self.default_canvas))
        self.canvas = FigureCanvas(self.figure)
        
        self.lb_info_process.setText(self.cbProcess.currentText())
        self.cbProcess.currentIndexChanged.connect(self.update_label_values)
        self.cbAge.currentIndexChanged.connect(self.update_label_values)
        self.cbGender.currentIndexChanged.connect(self.update_label_values)
        self.cbSound.currentIndexChanged.connect(self.update_label_values)
        self.cbClass.currentIndexChanged.connect(self.update_label_values)

        
        self.radioButtonViewRawData.toggled.connect(self.spectrum)
        
        self.pb_PlayButton.clicked.connect(self.play_audio)

        self.pbRemoveButton.clicked.connect(self.removeItemfromlist)

        self.pb_audio_editor.clicked.connect(self.connectEditor)
        
        self.cb_Class__.currentIndexChanged.connect(self.update_actionBox)
        self.cb_Class__.setCurrentIndex(0)
        self.update_actionBox(0)


        self.layout = QVBoxLayout(self.widget)
        self.layout.addWidget(self.canvas)

        self.widget.setStyleSheet("border: 1px solid blue;") 
        
        self.player = QMediaPlayer()
        
        self.data = None
        self.sr = 22050
        selected_sr = 0
        self.audio_file_path = None
        self.individual_audio = None
        self.individual_audio_append = None
        self.mfcc_data = []

    def connectEditor(self):
        self.editor_window = EditorWindow()
        self.editor_window.show()

    def update_time(self):
        current_time = QDateTime.currentDateTime()
        self.dtDateTime.setDateTime(current_time)

        

    def update_actionBox(self,index):
        print("Update action box called with index:", index)
        self.cbAction.clear()
        self.clearWidget()
        if index == 1:
            self.cbAction.addItems(['Append','New','View'])
            self.cbAction.setCurrentIndex(2)
            self.cbAction.currentIndexChanged.connect(self.update_actionValuesIndividual)
            self.update_actionValuesIndividual(2)
            self.cbProcess.setCurrentIndex(0)
            
        elif index ==0:
            self.cbAction.addItems(['Append','Analyse','View'])
            self.cbAction.setCurrentIndex(1)
            self.cbAction.currentIndexChanged.connect(self.update_actionValuesGroup)
            self.update_actionValuesGroup(1)
            self.cbProcess.setCurrentIndex(0)

        else:
            print("Invalid index:", index)
            

    def update_actionValuesGroup(self,index):
        print("Update action values group called with index:", index)
        
        self.clearWidget()

        if self.pbBrowse.receivers(self.pbBrowse.clicked) > 0:
            self.pbBrowse.clicked.disconnect()

        if self.pb_AnalyzeButton.receivers(self.pb_AnalyzeButton.clicked) > 0:
            self.pb_AnalyzeButton.clicked.disconnect()

        if index == 2:
            self.pb_AnalyzeButton.setText('View Data')
            self.labelSourceName.setVisible(False)
            self.pbBrowse.setVisible(False)
            self.lbl_File_Folder_Name.setVisible(False)
            self.pb_AnalyzeButton.setEnabled(True)
            self.pb_AnalyzeButton.setVisible(True)

            self.grpStatistics.setEnabled(False)
            self.gridGroupBox.setEnabled(False)
            self.lbSampleFreqValue.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.lbDemonSpan.setVisible(False)
            self.lbDemonSpanValue.setVisible(False)
            self.dsbDemonSpan.setVisible(False)
            self.label_NewPatientId.setVisible(False)
            self.txtEdit_NewPatientId.setVisible(False)
            self.label_Error.setVisible(False)
            self.label_ErrorData.setVisible(False)
            self.label_PatientSearch.setVisible(False)
            self.txtEdit_PatientSearch.setVisible(False)
            self.label_SelectPatientId.setVisible(False)
            self.lstWgt_PatientList.setVisible(False)
            self.label_DateTime.setVisible(False)
            self.dtDateTime.setVisible(False)
            self.label_Error_2.setVisible(False)
            self.label_ErrorDateTime.setVisible(False)
            self.progressBar.setVisible(False)
            self.pbSaveButton.setVisible(False)
            self.pbDiscardButton.setVisible(False)
            self.pbSaveAllButton.setVisible(False)
            self.pbDiscardAllButton.setVisible(False)
            self.pb_SaveRawData.setVisible(False)
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.pb_PlayButton.setVisible(False)
        
            self.label_radioButtonView.setVisible(False)
            self.radioButtonViewHistogram.setVisible(False)
            self.radioButtonViewRawData.setVisible(False)

            self.groupBox_GraphInfo.setVisible(True)
            self.grpView.setVisible(True)
            self.labelAge.setVisible(True)
            self.cbAge.setVisible(True)
            self.labelGender.setVisible(True)
            self.cbGender.setVisible(True)
            self.label_Class.setVisible(True)
            self.cbClass.setVisible(True)
            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)
            
        elif index == 1:

            #self.cbProcess.setCurrentIndex(1)
            self.cbProcess.currentIndexChanged.connect(self.group_analyzeProcess)
            self.pbBrowse.clicked.connect(self.openFile)
            self.pb_AnalyzeButton.clicked.connect(self.progressbar_update)
            self.labelSourceName.setText('Source File')
            self.lbl_File_Folder_Name.setText('Select a File')
            self.pb_AnalyzeButton.setText('Analyze')


            self.labelSourceName.setVisible(True)
            self.pbBrowse.setVisible(True)
            self.lbl_File_Folder_Name.setVisible(True)
            self.pb_AnalyzeButton.setEnabled(False)

            self.grpStatistics.setEnabled(False)
            self.gridGroupBox.setEnabled(False)
            self.lbSampleFreqValue.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.lbDemonSpan.setVisible(False)
            self.lbDemonSpanValue.setVisible(False)
            self.dsbDemonSpan.setVisible(False)
            self.label_NewPatientId.setVisible(False)
            self.txtEdit_NewPatientId.setVisible(False)
            self.label_Error.setVisible(False)
            self.label_ErrorData.setVisible(False)
            self.label_PatientSearch.setVisible(False)
            self.txtEdit_PatientSearch.setVisible(False)
            self.label_SelectPatientId.setVisible(False)
            self.lstWgt_PatientList.setVisible(False)
            self.label_DateTime.setVisible(False)
            self.dtDateTime.setVisible(False)
            self.label_Error_2.setVisible(False)
            self.label_ErrorDateTime.setVisible(False)
            self.progressBar.setVisible(False)
            self.pbSaveButton.setVisible(False)
            self.pbDiscardButton.setVisible(False)
            self.pbSaveAllButton.setVisible(False)
            self.pbDiscardAllButton.setVisible(False)
            self.pb_SaveRawData.setVisible(False)
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.pb_PlayButton.setVisible(False)
        
            self.label_radioButtonView.setVisible(False)
            self.radioButtonViewHistogram.setVisible(False)
            self.radioButtonViewRawData.setVisible(False)

            self.groupBox_GraphInfo.setVisible(True)
            self.grpView.setVisible(True)
            self.labelAge.setVisible(True)
            self.cbAge.setVisible(True)
            self.labelGender.setVisible(True)
            self.cbGender.setVisible(True)
            self.label_Class.setVisible(True)
            self.cbClass.setVisible(True)
            self.cbSampleFreq.setVisible(True)
            self.label_View.setVisible(False)
            self.cbView.setVisible(False)

        elif index == 0:
            
            self.labelSourceName.setText('Source Folder')
            self.lbl_File_Folder_Name.setText('Select a Folder')
            self.pb_AnalyzeButton.setText('Extract Feature')

            self.pbBrowse.clicked.connect(self.openFolder)

            
            self.labelSourceName.setVisible(True)
            self.pbBrowse.setVisible(True)
            self.lbl_File_Folder_Name.setVisible(True)
            self.pb_AnalyzeButton.setEnabled(False)

            self.grpStatistics.setEnabled(False)
            self.gridGroupBox.setEnabled(False)
            self.lbSampleFreqValue.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.lbDemonSpan.setVisible(False)
            self.lbDemonSpanValue.setVisible(False)
            self.dsbDemonSpan.setVisible(False)
            self.label_NewPatientId.setVisible(False)
            self.txtEdit_NewPatientId.setVisible(False)
            self.label_Error.setVisible(False)
            self.label_ErrorData.setVisible(False)
            self.label_PatientSearch.setVisible(False)
            self.txtEdit_PatientSearch.setVisible(False)
            self.label_SelectPatientId.setVisible(False)
            self.lstWgt_PatientList.setVisible(False)
            self.label_DateTime.setVisible(False)
            self.dtDateTime.setVisible(False)
            self.label_Error_2.setVisible(False)
            self.label_ErrorDateTime.setVisible(False)
            self.progressBar.setVisible(False)
            self.pbSaveButton.setVisible(False)
            self.pbDiscardButton.setVisible(False)
            self.pbSaveAllButton.setVisible(False)
            self.pbDiscardAllButton.setVisible(False)
            self.pb_SaveRawData.setVisible(False)
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.pb_PlayButton.setVisible(False)
        
            self.label_radioButtonView.setVisible(False)
            self.radioButtonViewHistogram.setVisible(False)
            self.radioButtonViewRawData.setVisible(False)
            self.groupBox_GraphInfo.setVisible(True)
            self.grpView.setVisible(True)
            self.labelAge.setVisible(True)
            self.cbAge.setVisible(True)
            self.labelGender.setVisible(True)
            self.cbGender.setVisible(True)
            self.label_Class.setVisible(True)
            self.cbClass.setVisible(True)
            self.cbSampleFreq.setVisible(True)
            
        else:
             print("Invalid index:", index)

    def update_actionValuesIndividual(self,index):
        print("Update action values individual called with index:", index)

        if self.pbBrowse.receivers(self.pbBrowse.clicked) > 0:
            self.pbBrowse.clicked.disconnect()

        if self.pb_AnalyzeButton.receivers(self.pb_AnalyzeButton.clicked) > 0:
            self.pb_AnalyzeButton.clicked.disconnect()

            
        if index == 2:
            self.clearWidget()
            
            self.label_PatientSearch.setVisible(True)
            self.txtEdit_PatientSearch.setVisible(True)
            self.label_SelectPatientId.setVisible(True)
            self.lstWgt_PatientList.setVisible(True)
                    
            self.labelAge.setVisible(False)
            self.cbAge.setVisible(False)
            self.labelGender.setVisible(False)
            self.cbGender.setVisible(False)
            self.label_Class.setVisible(False)
            self.cbClass.setVisible(False)
            self.pbImportDbButton.setVisible(True)
            self.groupBox_GraphInfo.setVisible(False)
            self.pb_AnalyzeButton.setVisible(False)
            self.pb_SaveRawData.setVisible(False)
            self.grpView.setVisible(False)
            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.label_radioButtonView.setVisible(False)
            self.radioButtonViewHistogram.setVisible(False)
            self.radioButtonViewRawData.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.pb_ApplyNrmValues.setEnabled(False)
            self.gridGroupBox.setEnabled(False)
            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)
            self.lbDemonSpan.setVisible(True)
            self.lbDemonSpanValue.setVisible(True)
            self.label_Error_2.setVisible(False)
            self.label_ErrorDateTime.setVisible(False)
            self.pushButtonForPdf.setEnabled(True)
            self.pb_ViewPdf.setEnabled(True)

            self.cbProcess.currentIndexChanged.connect(self.individual_analyzeprocess)

            

        elif index == 1:
            
            self.clearWidget()
            
            self.label_PatientSearch.setVisible(False)
            self.txtEdit_PatientSearch.setVisible(False)
            self.label_SelectPatientId.setVisible(False)
            self.lstWgt_PatientList.setVisible(False)
            self.label_NewPatientId.setVisible(True)
            self.txtEdit_NewPatientId.setVisible(True)
            self.label_DateTime.setVisible(True)
            self.dtDateTime.setVisible(True)
            self.label_Error_2.setVisible(False)
            self.label_ErrorDateTime.setVisible(False)
            self.labelSourceName.setText('Source File')
            self.lbl_File_Folder_Name.setText('Select a File')
            self.pb_AnalyzeButton.setText('Extract Feature')
            self.labelSourceName.setVisible(True)
            self.pbBrowse.setVisible(True)
            self.lbl_File_Folder_Name.setVisible(True)
            self.pb_AnalyzeButton.setVisible(True)
            self.pb_AnalyzeButton.setEnabled(False)
            self.lbDemonSpanValue.setVisible(False)
            self.dsbDemonSpan.setVisible(True)
            self.lbSampleFreqValue.setVisible(False)
            self.cbSampleFreq.setVisible(True)
            self.norm_groupBox.setVisible(False)
            self.grpView.setVisible(False)
            self.labelAge.setVisible(False)
            self.cbAge.setVisible(False)
            self.labelGender.setVisible(False)
            self.cbGender.setVisible(False)
            self.label_Class.setVisible(False)
            self.cbClass.setVisible(False)
            self.groupBox_GraphInfo.setVisible(False)
            self.pushButtonForPdf.setEnabled(True)
            self.pb_ViewPdf.setEnabled(True)
            
            self.pbBrowse.clicked.connect(self.openFile_individual)
            self.pb_AnalyzeButton.clicked.connect(self.progressbar_update_individual)
            #self.cbProcess.currentIndexChanged.connect(self.individual_analyzeprocess)


        else:
            #self.clearWidget()
            self.label_NewPatientId.setVisible(False)
            self.txtEdit_NewPatientId.setVisible(False)
            self.dsbDemonSpan.setVisible(False)
            self.labelSourceName.setText('Source File')
            self.lbl_File_Folder_Name.setText('Select a File')
            self.pb_AnalyzeButton.setText('Extract Feature')
            self.pb_AnalyzeButton.setVisible(True)
            self.pb_AnalyzeButton.setEnabled(False)
            self.label_PatientSearch.setVisible(True)
            self.txtEdit_PatientSearch.setVisible(True)
            self.label_SelectPatientId.setVisible(True)
            self.lstWgt_PatientList.setVisible(True)
            self.label_DateTime.setVisible(True)
            self.dtDateTime.setVisible(True)
            self.lbSampleFreqValue.setVisible(True)
            self.lbDemonSpan.setVisible(True)
            self.lbDemonSpanValue.setVisible(True)
            self.cbSampleFreq.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.grpView.setVisible(False)
            self.labelAge.setVisible(False)
            self.cbAge.setVisible(False)
            self.labelGender.setVisible(False)
            self.cbGender.setVisible(False)
            self.label_Class.setVisible(False)
            self.cbClass.setVisible(False)
            self.groupBox_GraphInfo.setVisible(False)
            self.pushButtonForPdf.setEnabled(True)
            self.pb_ViewPdf.setEnabled(True)

            self.pbBrowse.clicked.connect(self.openFile_appendIndividual)
            self.pb_AnalyzeButton.clicked.connect(self.append_individual_process)
           


    def update_label_values(self):
        selected_process = self.cbProcess.currentText()
        selected_age = self.cbAge.currentText()
        selected_gender = self.cbGender.currentText()
        selected_sound = self.cbSound.currentText()
        selected_class = self.cbClass.currentText()

        self.lb_info_process.setText(f'{selected_process}')
        self.lb_info_age.setText(f'{selected_age}')
        self.lb_info_gender.setText(f'{selected_gender}')
        self.lb_info_sound.setText(f'{selected_sound}')
        self.lb_info_class.setText(f'{selected_class}')

    
    def group_analyzeProcess(self,index):

        if self.pb_AnalyzeButton.receivers(self.pb_AnalyzeButton.clicked) > 0:
            self.pb_AnalyzeButton.clicked.disconnect()
        
        self.lst_Analyze.clear()
        self.clearWidget()
        self.pushButtonForPdf.setEnabled(False)
        self.pb_ViewPdf.setEnabled(False)
        if index == 0:
            self.pbBrowse.clicked.connect(self.openFile)
            self.pb_AnalyzeButton.clicked.connect(self.progressbar_update)
            
            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.cbSampleFreq.setVisible(True)
            self.lbSampleFreqValue.setVisible(False)
            self.grpStatistics.setEnabled(False)
            #self.cbView.currentIndexChanged.connect(self.processCombo)

        elif index == 1:
            self.clearWidget()
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.pb_AnalyzeButton.clicked.connect(self.updateHistogram_bins)
            self.pbBinOkButton.clicked.connect(self.updateHistogram_bins)
            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)
            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.norm_groupBox.setVisible(False)
            
            

        elif index == 2:
            self.clearWidget()
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.progressBar.setVisible(False)
            self.lst_Analyze.reset()
            self.pb_AnalyzeButton.clicked.connect(self.spectrum)

            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.norm_groupBox.setVisible(False)
            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)
            self.grpStatistics.setEnabled(False)

        elif index == 3:
            self.clearWidget()
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.progressBar.setVisible(False)
            self.lst_Analyze.reset()
            self.pb_AnalyzeButton.clicked.connect(self.cepstrum)

            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.norm_groupBox.setVisible(True)
            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)
            self.grpStatistics.setEnabled(False)

        elif index == 6:
            self.clearWidget()
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.progressBar.setVisible(False)
            self.lst_Analyze.reset()

            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.grpStatistics.setEnabled(True)
            self.norm_groupBox.setVisible(False)
            self.pb_AnalyzeButton.clicked.connect(self.processCentroid)
            self.grpStatistics_mean.toggled.connect(lambda:self.spectralCentroid_statistics('mean'))
            self.grpStatistics_std.toggled.connect(lambda: self.spectralCentroid_statistics('std'))
            self.grpStatistics_min.toggled.connect(lambda: self.spectralCentroid_statistics('min'))
            self.grpStatistics_max.toggled.connect(lambda: self.spectralCentroid_statistics('max'))
            self.grpStatistics_median.toggled.connect(lambda: self.spectralCentroid_statistics('median'))
            self.grpStatistics_maxbymean.toggled.connect(lambda: self.spectralCentroid_statistics('max_by_mean'))
            self.grpStatistics_maxbymedian.toggled.connect(lambda: self.spectralCentroid_statistics('max_by_median'))
            self.grpStatistics_stdbymean.toggled.connect(lambda: self.spectralCentroid_statistics('std_by_mean'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('mean'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('std'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('min'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('max'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('median'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('max_by_mean'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('max_by_median'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralCentroid_statistics('std_by_mean'))

            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)

        elif index == 7:
            self.clearWidget()
            self.lst_Analyze.setVisible(False)
            self.pbAppendButton.setVisible(False)
            self.pbRemoveButton.setVisible(False)
            self.progressBar.setVisible(False)
            self.lst_Analyze.reset()

            self.label_View.setVisible(False)
            self.cbView.setVisible(False)
            self.grpStatistics.setEnabled(True)
            self.norm_groupBox.setVisible(False)
            self.pb_AnalyzeButton.clicked.connect(self.processSpread)
            self.grpStatistics_mean.toggled.connect(lambda:self.spectralSpread_statistics('mean'))
            self.grpStatistics_std.toggled.connect(lambda: self.spectralSpread_statistics('std'))
            self.grpStatistics_min.toggled.connect(lambda: self.spectralSpread_statistics('min'))
            self.grpStatistics_max.toggled.connect(lambda: self.spectralSpread_statistics('max'))
            self.grpStatistics_median.toggled.connect(lambda: self.spectralSpread_statistics('median'))
            self.grpStatistics_maxbymean.toggled.connect(lambda: self.spectralSpread_statistics('max_by_mean'))
            self.grpStatistics_maxbymedian.toggled.connect(lambda: self.spectralSpread_statistics('max_by_median'))
            self.grpStatistics_stdbymean.toggled.connect(lambda: self.spectralSpread_statistics('std_by_mean'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('mean'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('std'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('min'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('max'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('median'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('max_by_mean'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('max_by_median'))
            self.pbBinOkButton.clicked.connect(lambda: self.spectralSpread_statistics('std_by_mean'))

            self.cbSampleFreq.setVisible(False)
            self.lbSampleFreqValue.setVisible(True)

    

    def individual_analyzeprocess(self,index):
        self.gridGroupBox.setEnabled(True)
        if index == 1:
            self.mfcc_individual_initial()

        elif index == 2:
            self.spectrum_individual()
            
        elif index == 3:
            self.cepstrum_individual()
        elif index == 6:
            self.grpStatistics.setEnabled(True)
            self.spectralCentroid_individual_initial()
        elif index == 7:
            self.spectralSpread_individual_initial()

    '''
    def append_individual_process(self):
        index = self.cbProcess.currentIndex()
        selected_process = self.cbProcess.itemText(index)
        if selected_process == "MFCC":
            ax = self.mfcc_individual_initial()
            self.appendMFCC_individual(ax.get_xlim())
        elif selected_process == "Spectral Centroid":
            self.grpStatistics.setEnabled(True)
            ax = self.spectralCentroid_individual_initial()
            self.spectralCentroid_individual_append(ax.get_xlim())
        
            self.grpStatistics_mean.toggled.connect(lambda:self.spectralCentroid_individual_append('mean'))
            self.grpStatistics_std.toggled.connect(lambda: self.spectralCentroid_individual_append('std'))
            self.grpStatistics_min.toggled.connect(lambda: self.spectralCentroid_individual_append('min'))
            self.grpStatistics_max.toggled.connect(lambda: self.spectralCentroid_individual_append('max'))
            self.grpStatistics_median.toggled.connect(lambda: self.spectralCentroid_individual_append('median'))
    '''
    def append_individual_process(self):
        index = self.cbProcess.currentIndex()
        selected_process = self.cbProcess.itemText(index)
        if selected_process == "MFCC":
            ax = self.mfcc_individual_initial()
            self.appendMFCC_individual(ax.get_xlim())
        elif selected_process == "Spectral Centroid":
            self.grpStatistics.setEnabled(True)
            ax = self.spectralCentroid_individual_initial()
            x_lim = ax.get_xlim()
            self.spectralCentroid_individual_append(statistic='mean', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='median', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='max', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='min', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='std', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='max_by_mean', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='max_by_median', x_lim=x_lim)
            self.spectralCentroid_individual_append(statistic='std_by_mean', x_lim=x_lim)

            self.disconnect_radio_button_signals()

            self.grpStatistics_mean.toggled.connect(lambda:self.spectralCentroid_individual_append('mean', x_lim=x_lim))
            self.grpStatistics_std.toggled.connect(lambda: self.spectralCentroid_individual_append('std', x_lim=x_lim))
            self.grpStatistics_min.toggled.connect(lambda: self.spectralCentroid_individual_append('min', x_lim=x_lim))
            self.grpStatistics_max.toggled.connect(lambda: self.spectralCentroid_individual_append('max', x_lim=x_lim))
            self.grpStatistics_median.toggled.connect(lambda: self.spectralCentroid_individual_append('median', x_lim=x_lim))
            self.grpStatistics_maxbymean.toggled.connect(lambda: self.spectralCentroid_individual_append('max_by_mean', x_lim=x_lim))
            self.grpStatistics_maxbymedian.toggled.connect(lambda: self.spectralCentroid_individual_append('max_by_median', x_lim=x_lim))
            self.grpStatistics_stdbymean.toggled.connect(lambda: self.spectralCentroid_individual_append('std_by_mean', x_lim=x_lim))
                      
    def openFile(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Wave Files(*.wav)", options=options)

        if fileName:
            print(fileName)
           
            self.lbl_File_Folder_Name.setText(fileName)
            self.audio_file_path = fileName
            self.pb_AnalyzeButton.setEnabled(True)
            self.pb_PlayButton.setVisible(True)

    
    def openFolder(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        folderPath = QFileDialog.getExistingDirectory(self, "select folder containing WAV files", options=options)

        if folderPath:
            print(folderPath)

            self.lbl_File_Folder_Name.setText(folderPath)
            self.pb_AnalyzeButton.setEnabled(True)
            
    def openFile_individual(self):

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Wave Files(*.wav)", options=options)

        if fileName:
            print(fileName)
           
            self.lbl_File_Folder_Name.setText(fileName)
            self.individual_audio = fileName

            self.pb_AnalyzeButton.setEnabled(True)
            self.pb_PlayButton.setVisible(True)

    def openFile_appendIndividual(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "Wave Files(*.wav)", options=options)

        if fileName:
            print(fileName)
           
            self.lbl_File_Folder_Name.setText(fileName)
            self.individual_audio_append = fileName

            self.pb_AnalyzeButton.setEnabled(True)
            self.pb_PlayButton.setVisible(True)

    def updateHistogram_bins(self):

        self.grpStatistics.setEnabled(False)
        self.gridGroupBox.setEnabled(True)
        self.norm_groupBox.setVisible(False)
        
        self.sr,self.data = read(self.audio_file_path)
        
        duration = len(self.data) / self.sr

        time = np.arange(0, duration, 1 / self.sr)

        self.canvas.figure.clear()

        self.canvas.figure.set_size_inches(self.default_canvas)
        
        ax = self.canvas.figure.add_subplot(111)

        
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))
        
        mfcc_features = mfcc(self.data, self.sr, nfft=4096)

        no_of_bins = self.le_NumOfBins.text()
        bin_number = int(no_of_bins)

        mfcc_distances = np.sqrt(np.sum(np.diff(mfcc_features, axis=0)**2, axis=1))
        counts, bins, _ = ax.hist(mfcc_distances, bins=bin_number, color='white', alpha=0.7, edgecolor='black')

        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        
        ax.set_xlabel('MFCC Distance')
        ax.set_ylabel('Number of Occurrences')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        #ax.xaxis.set_label_coords(0.5, -0.1)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(min(counts), max(counts))

        ax.spines['bottom'].set_bounds(bins[0], bins[-1])
      
        self.canvas.draw()

        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)

        self.lst_Analyze.setVisible(True)
        self.pbAppendButton.setVisible(True)
        self.pbRemoveButton.setVisible(True)

        filename=os.path.basename(self.audio_file_path)
        self.lst_Analyze.addItem(filename)

        self.pushButtonForPdf.clicked.connect(self.print_pdf)
        self.pb_ViewPdf.clicked.connect(self.view_pdf)
        self.cbSampleFreq.currentIndexChanged.connect(self.update_histogram)
        self.pbBinOkButton.clicked.connect(self.updateHistogram_bins)
        
    def mfcc_individual_initial(self):
        
        self.canvas.figure.clear()

        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)
        
        self.sr,self.data = read(self.individual_audio)
        

        pre_emphasis = 0.97
        emphasized_signal = np.append(self.data[0], self.data[1:] - pre_emphasis * self.data[:-1])

        frame_length = 0.025
        frame_stride = 0.01
        frame_length_samples = int(round(frame_length * self.sr))
        frame_stride_samples = int(round(frame_stride * self.sr))

        num_samples = len(emphasized_signal)
        num_frames = int(np.ceil(float(np.abs(num_samples - frame_length_samples)) / frame_stride_samples))

        pad_signal_length = num_frames * frame_stride_samples + frame_length_samples
        z = np.zeros((pad_signal_length - num_samples))
        pad_signal = np.append(emphasized_signal, z)

        indices = np.tile(np.arange(0, frame_length_samples), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_stride_samples, frame_stride_samples), (frame_length_samples, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length_samples)

        mag_frames = np.absolute(np.fft.rfft(frames, 4096))
        pow_frames = ((1.0 / 4096) * ((mag_frames) ** 2))

        low_freq_mel = 0
        if 22050 / 2:
            high_freq_mel = (2595 * np.log10(1 + (self.sr / 2) / 700))
        else:
            high_freq_mel = (2595 * np.log10(1 + (22050 / 2) / 700))

        mel_points = np.linspace(low_freq_mel, high_freq_mel, 32 + 2)
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))

        bin_points = np.floor((4096 + 1) * hz_points / self.sr)

        filter_banks = np.zeros((32, int(np.floor(4096 / 2 + 1))))
        for m in range(1, 32 + 1):
            f_m_minus = int(bin_points[m - 1])
            f_m = int(bin_points[m])
            f_m_plus = int(bin_points[m + 1])

            for k in range(f_m_minus, f_m):
                filter_banks[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                filter_banks[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filtered_spectrogram = np.dot(pow_frames, filter_banks.T)
        filtered_spectrogram = np.where(filtered_spectrogram == 0, np.finfo(float).eps, filtered_spectrogram)

        filtered_spectrogram = 10 * np.log10(filtered_spectrogram)

        mfcc = dct(filtered_spectrogram, type=2, axis=1, norm='ortho')[:, 1:(13 + 1)]


        mfcc_positive = np.maximum(mfcc, 0)

        mean_mfcc = np.mean(mfcc_positive)

        print("Mean MFCC value (positive values only):")
        print(mean_mfcc)

        self.canvas.figure.set_size_inches(self.default_canvas)
        ax = self.canvas.figure.add_subplot(111)
        
        #ax.plot(mean_mfcc, 'bo')
        ax.plot(0, 100, 'bo')

        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        ax.set_xlim(-0.015,(mean_mfcc) - 1)
        ax.get_xaxis().set_visible(False)
        y_ticks = np.arange(0.00, 201, 20.00)
        ax.set_yticks(y_ticks)
        self.canvas.draw()

        return ax

        sound_directory = self.cbSound.currentText()
        text = self.txtEdit_NewPatientId.toPlainText().strip()
        parent_directory = '_VoiceAnalyzerData'
        process_directory = self.cbProcess.currentText()
        category_directory = self.cb_Class__.currentText()
        sound_directory_path = os.path.join(parent_directory, process_directory, category_directory, text, sound_directory)
        code_directory = os.path.join(sound_directory_path, '_code')
        date_directory = os.path.join(sound_directory_path, '_date')
        mfcc_directory = os.path.join(sound_directory_path,'_mfcc')
        note_directory = os.path.join(sound_directory_path,'_note')
        
        os.makedirs(code_directory, exist_ok=True)
        os.makedirs(date_directory,exist_ok=True)
        os.makedirs(mfcc_directory,exist_ok=True)
        os.makedirs(note_directory,exist_ok=True)

        sampling_frequency = str(self.sr)
        sampling_frequency_path = os.path.join(sound_directory_path, '_samp_freq.txt')
        with open(sampling_frequency_path, 'w') as file:
            file.write(sampling_frequency)

        date_time = QDateTime.currentDateTime().toString("yyyy MM dd hh mm")

        current_date_time = os.path.join(date_directory, '_date.txt')
        with open(current_date_time, 'w') as file:
            file.write(date_time)
            
        self.pushButtonForPdf.clicked.connect(self.print_pdf_individual)
        self.pb_ViewPdf.clicked.connect(self.view_pdf_individual)

    

    def appendMFCC_individual(self,x_lim):
        
        self.sr, self.data = read(self.individual_audio_append)
        

        pre_emphasis = 0.97
        emphasized_signal = np.append(self.data[0], self.data[1:] - pre_emphasis * self.data[:-1])

        frame_length = 0.025
        frame_stride = 0.01
        frame_length_samples = int(round(frame_length * self.sr))
        frame_stride_samples = int(round(frame_stride * self.sr))

        num_samples = len(emphasized_signal)
        num_frames = int(np.ceil(float(np.abs(num_samples - frame_length_samples)) / frame_stride_samples))

        pad_signal_length = num_frames * frame_stride_samples + frame_length_samples
        z = np.zeros((pad_signal_length - num_samples))
        pad_signal = np.append(emphasized_signal, z)

        indices = np.tile(np.arange(0, frame_length_samples), (num_frames, 1)) + np.tile(
            np.arange(0, num_frames * frame_stride_samples, frame_stride_samples), (frame_length_samples, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        frames *= np.hamming(frame_length_samples)

        mag_frames = np.absolute(np.fft.rfft(frames, 4096))
        pow_frames = ((1.0 / 4096) * ((mag_frames) ** 2))

        low_freq_mel = 0
        if 22050 / 2:
            high_freq_mel = (2595 * np.log10(1 + (self.sr / 2) / 700))
        else:
            high_freq_mel = (2595 * np.log10(1 + (22050 / 2) / 700))

        mel_points = np.linspace(low_freq_mel, high_freq_mel, 32 + 2)
        hz_points = (700 * (10 ** (mel_points / 2595) - 1))

        bin_points = np.floor((4096 + 1) * hz_points / self.sr)

        filter_banks = np.zeros((32, int(np.floor(4096 / 2 + 1))))
        for m in range(1, 32 + 1):
            f_m_minus = int(bin_points[m - 1])
            f_m = int(bin_points[m])
            f_m_plus = int(bin_points[m + 1])

            for k in range(f_m_minus, f_m):
                filter_banks[m - 1, k] = (k - bin_points[m - 1]) / (bin_points[m] - bin_points[m - 1])
            for k in range(f_m, f_m_plus):
                filter_banks[m - 1, k] = (bin_points[m + 1] - k) / (bin_points[m + 1] - bin_points[m])

        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filtered_spectrogram = np.dot(pow_frames, filter_banks.T)
        filtered_spectrogram = np.where(filtered_spectrogram == 0, np.finfo(float).eps, filtered_spectrogram)

        filtered_spectrogram = 10 * np.log10(filtered_spectrogram)

        mfcc = dct(filtered_spectrogram, type=2, axis=1, norm='ortho')[:, 1:(13 + 1)]


        mfcc_positive = np.maximum(mfcc, 0)

        mean_mfcc = np.mean(mfcc_positive)

        print("Mean MFCC value (positive values only):")
        print(mean_mfcc)

        ax = self.canvas.figure.axes[0]
        
        xdistance = 0.030
        
        last_x = ax.lines[-1].get_xdata()[-1] if ax.lines else 0

        
        ax.plot(last_x + xdistance,mean_mfcc+100,'bo')


        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        ax.set_xlim(x_lim)
        ax.get_xaxis().set_visible(False)
        y_ticks = np.arange(0.00, 201, 20.00)
        ax.set_yticks(y_ticks)
        self.canvas.draw()
    
    
    def play_audio(self):

        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:    
            if self.audio_file_path:
                content = QMediaContent(QUrl.fromLocalFile(self.audio_file_path))
                self.player.setMedia(content)
                self.player.play()
        if self.player.state() == QMediaPlayer.PlayingState:
            self.pb_PlayButton.setIcon(QIcon('stop.png'))
        else:
            self.pb_PlayButton.setIcon(QIcon('play.png'))
            

    def progressbar_update(self):

        self.progressBar.setVisible(True)
        self.cb_Class__.setEnabled(False)
        self.cbProcess.setEnabled(False)
        self.cbAction.setEnabled(False)
        self.cbAge.setEnabled(False)
        self.cbGender.setEnabled(False)
        self.cbSound.setEnabled(False)
        self.cbClass.setEnabled(False)
        self.pbBrowse.setEnabled(False)
        self.pbImportDbButton.setEnabled(False)
  
        for i in range(101):  
            time.sleep(0.01) 
            self.progressBar.setValue(i)
            if i == 100:
                self.updateHistogram_bins()
                self.progressBar.reset()
                self.update_label_values()

                self.progressBar.setVisible(False)
                self.lst_Analyze.setVisible(True)
                self.pbAppendButton.setVisible(True)
                self.pbRemoveButton.setVisible(True)
                
                self.cb_Class__.setEnabled(True)
                self.cbProcess.setEnabled(True)
                self.cbAction.setEnabled(True)
                self.cbAge.setEnabled(True)
                self.cbGender.setEnabled(True)
                self.cbSound.setEnabled(True)
                self.cbClass.setEnabled(True)
                self.pbBrowse.setEnabled(True)
                self.pbImportDbButton.setEnabled(True)
                self.label_View.setVisible(True)
                self.cbView.setVisible(True)

    def progressbar_update_individual(self):
        self.progressBar.setVisible(True)

        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)

        for i in range(101):  
            time.sleep(0.01) 
            self.progressBar.setValue(i)
            if i == 100:
                self.create_directory()
                self.cbAction.setCurrentIndex(2)
                self.cbProcess.currentIndexChanged.connect(self.individual_analyzeprocess)

    def create_directory(self):
        text = self.txtEdit_NewPatientId.toPlainText().strip()
        parent_directory = '_VoiceAnalyzerData'
        process_directory = self.cbProcess.currentText()
        category_directory = self.cb_Class__.currentText()
        sound_directory = self.cbSound.currentText()
        
        if process_directory == "All":
            
            for index in range(1, self.cbProcess.count()):
                current_process = self.cbProcess.itemText(index)
                folder_path = os.path.join(parent_directory, current_process, category_directory, text, sound_directory)
                os.makedirs(folder_path)
                
                
                category_path = os.path.join(parent_directory, current_process, category_directory)
                if not os.path.exists(category_path):
                    os.makedirs(category_path)

               
                subdirectories = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
                for subdir in subdirectories:
                    if not self.lstWgt_PatientList.findItems(subdir, QtCore.Qt.MatchExactly):
                        self.lstWgt_PatientList.addItem(subdir)
        else:
            
            folder_path = os.path.join(parent_directory, process_directory, category_directory, text, sound_directory)
            os.makedirs(folder_path)

            
            category_path = os.path.join(parent_directory, process_directory, category_directory)
            if not os.path.exists(category_path):
                os.makedirs(category_path)

            
            subdirectories = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
            for subdir in subdirectories:
                if not self.lstWgt_PatientList.findItems(subdir, QtCore.Qt.MatchExactly):
                    self.lstWgt_PatientList.addItem(subdir)

        #self.lstWgt_PatientList.addItem(text)

        item = self.lstWgt_PatientList.findItems(text, QtCore.Qt.MatchExactly)
        if item:
            self.lstWgt_PatientList.setCurrentItem(item[0])



    

    def removeItemfromlist(self):

        selected_item = self.lst_Analyze.currentItem()
        if selected_item is not None:
            self.lst_Analyze.takeItem(self.lst_Analyze.row(selected_item))
            if self.lst_Analyze.count() == 0:
                self.lst_Analyze.setVisible(False)
                self.pbAppendButton.setVisible(False)
                self.pbRemoveButton.setVisible(False)

    def update_histogram(self):
        
        selected_sr = int(self.cbSampleFreq.currentText())
        data_resampled = self.data if self.sr == selected_sr else np.interp(np.arange(0, len(self.data), self.sr / selected_sr), np.arange(len(self.data)), self.data)
        self.canvas.figure.clear()
        self.canvas.figure.set_size_inches(self.default_canvas)
        ax = self.canvas.figure.add_subplot(111)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        mfcc_features = mfcc(data_resampled, selected_sr, nfft=4096)

        mfcc_distances = np.sqrt(np.sum(np.diff(mfcc_features, axis=0)**2, axis=1))

        no_of_bins = self.le_NumOfBins.text()
        bin_number = int(no_of_bins)

        counts, bins, _ = ax.hist(mfcc_distances, bins = bin_number, color='white', alpha=0.7, edgecolor='black')
        ax.set_xlabel('MFCC Distance')
        ax.set_ylabel('Number of Occurrences')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        #ax.xaxis.set_label_coords(0.5, -0.1)

        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        
        self.canvas.draw()

        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)

        

        self.pushButtonForPdf.clicked.connect(self.print_pdf)

    def spectrum(self):

        self.sr,self.data = read(self.audio_file_path)

        if self.data is None or self.sr == 0:
            print("Error: Data or sampling rate is not set.")
            return

        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)
        
    
        N = len(self.data)
        normalize = N / 2
        fourier = fft(self.data)

        sampling_rate =4012
        frequency_axis = fftfreq(N, d=1.0 / sampling_rate)
        norm_amplitude = np.abs(fourier) / normalize

        self.canvas.figure.clear()
        self.canvas.figure.set_size_inches(self.default_canvas)
        ax = self.canvas.figure.add_subplot(111)

        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        ax.plot(frequency_axis[:len(frequency_axis)//2], np.abs(norm_amplitude[:len(norm_amplitude)//2]))

        #num_bins = 100  
        #ax.hist(frequency_axis[:len(frequency_axis)//2], bins=num_bins, weights=np.abs(norm_amplitude[:len(norm_amplitude)//2]),color='white', alpha=0.7, edgecolor='black')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('amplitude')
        ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.set_xlim(0, max(frequency_axis))
        ax.set_ylim(0, max(norm_amplitude))

    
        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        
        self.canvas.draw()

        self.lst_Analyze.setVisible(True)
        self.pbAppendButton.setVisible(True)
        self.pbRemoveButton.setVisible(True)
        
        filename=os.path.basename(self.audio_file_path)
        self.lst_Analyze.addItem(filename)

        self.pushButtonForPdf.clicked.connect(self.print_pdf)
        self.pb_ViewPdf.clicked.connect(self.view_pdf)

    def spectrum_individual(self):

        self.sr,self.data = read(self.individual_audio)

        if self.data is None or self.sr == 0:
            print("Error: Data or sampling rate is not set.")
            return

        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)
        
    
        N = len(self.data)
        normalize = N / 2
        fourier = fft(self.data)

        sampling_rate =4012
        frequency_axis = fftfreq(N, d=1.0 / sampling_rate)
        norm_amplitude = np.abs(fourier) / normalize

        self.canvas.figure.clear()
        self.canvas.figure.set_size_inches(self.default_canvas)
        ax = self.canvas.figure.add_subplot(111)

        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        ax.plot(frequency_axis[:len(frequency_axis)//2], np.abs(norm_amplitude[:len(norm_amplitude)//2]))
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('amplitude')
        ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.set_xlim(0, max(frequency_axis))
        ax.set_ylim(0, max(norm_amplitude))

    
        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        
        self.canvas.draw()
        
        #filename=os.path.basename(self.audio_file_path)
        #self.lst_Analyze.addItem(filename)

        self.pushButtonForPdf.clicked.connect(self.print_pdf_individual)
        self.pb_ViewPdf.clicked.connect(self.view_pdf_individual)

        sound_directory = self.cbSound.currentText()
        text = self.txtEdit_NewPatientId.toPlainText().strip()
        parent_directory = '_VoiceAnalyzerData'
        process_directory = self.cbProcess.currentText()
        category_directory = self.cb_Class__.currentText()
        sound_directory_path = os.path.join(parent_directory, process_directory, category_directory, text, sound_directory)
        date_directory = os.path.join(sound_directory_path, '_date')
        feature_directory = os.path.join(sound_directory_path,'_feature')
        note_directory = os.path.join(sound_directory_path,'_note')
        
        os.makedirs(date_directory, exist_ok=True)
        os.makedirs(feature_directory,exist_ok=True)
        os.makedirs(note_directory,exist_ok=True)

        sampling_frequency = str(sampling_rate)
        sampling_frequency_path = os.path.join(sound_directory_path, '_samp_freq.txt')
        with open(sampling_frequency_path, 'w') as file:
            file.write(sampling_frequency)

        date_time = QDateTime.currentDateTime().toString("yyyy MM dd hh mm")

        current_date_time = os.path.join(date_directory, '_date.txt')
        with open(current_date_time, 'w') as file:
            file.write(date_time)

    def print_pdf(self,filename):

        self.canvas.figure.set_size_inches(8.26, 11.69)

        ax = self.canvas.figure.axes[0]
        ax.set_position([0.1, 0.78, 0.36, 0.13])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.xaxis.label.set_horizontalalignment('right')
        ax.xaxis.label.set_y(-0.15)
        ax.xaxis.label.set_x(1)
        ax.xaxis.label.set_fontsize(8)
        

        ax.yaxis.label.set_horizontalalignment('right')
        ax.yaxis.label.set_x(-0.15)
        ax.yaxis.label.set_y(1)
        ax.yaxis.label.set_fontsize(8)

        now = datetime.datetime.now()

        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF","", "PDF Files (*.pdf)")
        if filename:
            try:
                with PdfPages(filename) as pdf:
                    num_pages = 1

                    selected_process = self.lb_info_process.text()
                    selected_age = self.lb_info_age.text()
                    selected_gender = self.lb_info_gender.text()
                    selected_sound = self.lb_info_sound.text()
                    selected_class = self.lb_info_class.text()

                    for i in range(num_pages):
                        self.canvas.figure.texts = []
                        self.canvas.figure.text(0.1, 0.95, f"Process : {selected_process}", fontsize=12, ha='left')
                        self.canvas.figure.text(0.5, 0.95, f"Gender  : {selected_gender}", fontsize=12, ha='center')
                        self.canvas.figure.text(0.9, 0.95, f"Class   : {selected_class}", fontsize=12, ha='right')
                        self.canvas.figure.text(0.1, 0.93, f"Age       : {selected_age}", fontsize=12, ha='left')
                        self.canvas.figure.text(0.5, 0.93, f"Sound    : {selected_sound}", fontsize=12, ha='center')

                        self.canvas.figure.text(0.93, 0.05, f"Report on {now.strftime('%d %B %Y %I:%M %p')}", fontsize=10, ha='right')
                        self.canvas.figure.text(0.07, 0.05, f"Page {i+1} of {num_pages}", fontsize=10, ha='left')

                        self.canvas.figure.text(0.05, 0.92, '─' * 90, fontsize=10, ha='left', va='center')
                        self.canvas.figure.text(0.05, 0.07, '─' * 90, fontsize=10, ha='left', va='center')

                        pdf.savefig(self.canvas.figure)
                        print(f"PDF saved as: {filename}")

                QDesktopServices.openUrl(QUrl.fromLocalFile(filename))
            except Exception as e:
                print(f"Error saving PDF: {e}")

    def view_pdf(self, filename):

        self.canvas.figure.set_size_inches(8.26, 11.69)

        ax = self.canvas.figure.axes[0]
        ax.set_position([0.1, 0.78, 0.36, 0.13])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.xaxis.label.set_horizontalalignment('right')
        ax.xaxis.label.set_y(-0.15)
        ax.xaxis.label.set_x(1)
        ax.xaxis.label.set_fontsize(8)
        

        ax.yaxis.label.set_horizontalalignment('right')
        ax.yaxis.label.set_x(-0.15)
        ax.yaxis.label.set_y(1)
        ax.yaxis.label.set_fontsize(8)

        now = datetime.datetime.now()
        filename = f"view.pdf"
        
        
        with PdfPages(filename) as pdf:

            num_pages = 1
            
            selected_process = self.lb_info_process.text()
            selected_age = self.lb_info_age.text()
            selected_gender = self.lb_info_gender.text()
            selected_sound = self.lb_info_sound.text()
            selected_class = self.lb_info_class.text()
            
            for i in range(num_pages):

                self.canvas.figure.texts = []

                self.canvas.figure.text(0.07, 0.97, f"Process : {selected_process}", fontsize=12, ha='left')
                self.canvas.figure.text(0.5, 0.97, f"Gender  : {selected_gender}", fontsize=12, ha='center')
                self.canvas.figure.text(0.9, 0.97, f"Class   : {selected_class}", fontsize=12, ha='right')
                self.canvas.figure.text(0.07, 0.94, f"Age       : {selected_age}", fontsize=12, ha='left')
                self.canvas.figure.text(0.5, 0.94, f"Sound    : {selected_sound}", fontsize=12, ha='center')
                self.canvas.figure.text(0.93, 0.05, f"Report on {now.strftime('%d %B %Y %I:%M %p')}", fontsize=10, ha='right')
                self.canvas.figure.text(0.07, 0.05, f"Page {i+1} of {num_pages}", fontsize=10, ha='left')

                self.canvas.figure.text(0.05, 0.92, '─' * 90, fontsize=10, ha='left', va='center')
                self.canvas.figure.text(0.05, 0.07, '─' * 90, fontsize=10, ha='left', va='center')
                
                pdf.savefig(self.canvas.figure)
                
            QDesktopServices.openUrl(QUrl.fromLocalFile(filename))

    def print_pdf_individual(self, filename):

        self.canvas.figure.set_size_inches(8.26, 11.69)

        ax = self.canvas.figure.axes[0]
        ax.set_position([0.1, 0.78, 0.36, 0.13])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.xaxis.label.set_horizontalalignment('right')
        ax.xaxis.label.set_y(-0.15)
        ax.xaxis.label.set_x(1)
        ax.xaxis.label.set_fontsize(8)
        

        ax.yaxis.label.set_horizontalalignment('right')
        ax.yaxis.label.set_x(-0.15)
        ax.yaxis.label.set_y(1)
        ax.yaxis.label.set_fontsize(8)

        now = datetime.datetime.now()
        filename, _ = QFileDialog.getSaveFileName(self, "Save PDF","", "PDF Files (*.pdf)")
        if filename:
            try:
                with PdfPages(filename) as pdf:
                    num_pages = 1

                    selected_patient_item = self.lstWgt_PatientList.currentItem()
                    selected_patient = selected_patient_item.text()
                    selected_process = self.lb_info_process.text()
                    selected_sound = self.lb_info_sound.text()

                    for i in range(num_pages):
                        self.canvas.figure.texts = []

                        self.canvas.figure.text(0.07, 0.97, f"Patient ID : {selected_patient}", fontsize=12, ha='left')
                        self.canvas.figure.text(0.5, 0.97, f"Process  : {selected_process}", fontsize=12, ha='center')
                        self.canvas.figure.text(0.5, 0.94, f"Sound    : {selected_sound}", fontsize=12, ha='center')

                        self.canvas.figure.text(0.93, 0.05, f"Report on {now.strftime('%d %B %Y %I:%M %p')}", fontsize=10, ha='right')
                        self.canvas.figure.text(0.07, 0.05, f"Page {i+1} of {num_pages}", fontsize=10, ha='left')

                        self.canvas.figure.text(0.05, 0.92, '─' * 90, fontsize=10, ha='left', va='center')
                        self.canvas.figure.text(0.05, 0.07, '─' * 90, fontsize=10, ha='left', va='center')

                        pdf.savefig(self.canvas.figure)
                        print(f"PDF saved as: {filename}")

                    QDesktopServices.openUrl(QUrl.fromLocalFile(filename))
            except Exception as e:
                print(f"Error saving PDF: {e}")         

    def view_pdf_individual(self, filename):

        self.canvas.figure.set_size_inches(8.26, 11.69)

        ax = self.canvas.figure.axes[0]
        ax.set_position([0.1, 0.78, 0.36, 0.13])
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))

        ax.xaxis.label.set_horizontalalignment('right')
        ax.xaxis.label.set_y(-0.15)
        ax.xaxis.label.set_x(1)
        ax.xaxis.label.set_fontsize(8)
        

        ax.yaxis.label.set_horizontalalignment('right')
        ax.yaxis.label.set_x(-0.15)
        ax.yaxis.label.set_y(1)
        ax.yaxis.label.set_fontsize(8)

        now = datetime.datetime.now()
        filename = f"view.pdf"
        
        
        with PdfPages(filename) as pdf:

            num_pages = 1
            
            selected_patient_item = self.lstWgt_PatientList.currentItem()
            selected_patient = selected_patient_item.text()
            selected_process = self.lb_info_process.text()
            selected_sound = self.lb_info_sound.text()
            
            for i in range(num_pages):

                self.canvas.figure.texts = []

                self.canvas.figure.text(0.07, 0.97, f"Patient ID : {selected_patient}", fontsize=12, ha='left')
                self.canvas.figure.text(0.5, 0.97, f"Process  : {selected_process}", fontsize=12, ha='center')
                self.canvas.figure.text(0.5, 0.94, f"Sound    : {selected_sound}", fontsize=12, ha='center')

                self.canvas.figure.text(0.93, 0.05, f"Report on {now.strftime('%d %B %Y %I:%M %p')}", fontsize=10, ha='right')
                self.canvas.figure.text(0.07, 0.05, f"Page {i+1} of {num_pages}", fontsize=10, ha='left')

                self.canvas.figure.text(0.05, 0.92, '─' * 90, fontsize=10, ha='left', va='center')
                self.canvas.figure.text(0.05, 0.07, '─' * 90, fontsize=10, ha='left', va='center')
                
                pdf.savefig(self.canvas.figure)
                
            QDesktopServices.openUrl(QUrl.fromLocalFile(filename))

    def cepstrum(self):

        self.sr,self.data = read(self.audio_file_path)
        
        if self.data is not None or self.sr == 0:
        
            n = len(self.data)
            normalize = n / 2
            fourier = fft(self.data)
            sampling_rate = self.sr
            
            log_spectrum = np.log(np.abs(fourier))
            cepstrum_result = np.fft.ifft(log_spectrum)
            
            self.canvas.figure.clear()
            self.canvas.figure.set_size_inches(self.default_canvas)
            self.ax = self.canvas.figure.add_subplot(111)

            self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            self.ax.yaxis.set_major_locator(plt.MaxNLocator(10))
            
            frequency_axis = fftfreq(n, d=1.0 / sampling_rate)
            norm_amplitude = np.abs(fourier) / normalize
            self.ax.plot(frequency_axis[:len(frequency_axis) // 2], norm_amplitude[:len(norm_amplitude) // 2], label='Spectrum')
            
            quefrency_axis = fftfreq(len(cepstrum_result), d=1.0 / sampling_rate)
            positive_part = np.abs(cepstrum_result)[:len(cepstrum_result) // 2]
            self.ax.plot(quefrency_axis[:len(quefrency_axis) // 2], positive_part, label='Cepstrum (Positive Part)')
            
            self.ax.set_xlabel('Frequency (Hz)')
            self.ax.set_ylabel('Amplitude')

            self.ax.yaxis.set_label_coords(-0.1, 0.5)
            #self.ax.xaxis.set_label_coords(0.5, -0.1)

            self.ax.set_xlim(0, max(quefrency_axis))
            self.ax.set_ylim(0, max(norm_amplitude))

            self.ax.spines['right'].set_color('blue')
            self.ax.spines['top'].set_color('blue')
            self.ax.spines['bottom'].set_color('blue')
            self.ax.spines['left'].set_color('blue')
            self.ax.xaxis.label.set_color('blue')
            self.ax.yaxis.label.set_color('blue')
            self.ax.xaxis.label.set_color('blue')
            self.ax.tick_params(axis='x', colors='blue')
            self.ax.tick_params(axis='y', colors='blue')
            self.ax.patch.set_edgecolor('blue')
            
            #self.canvas.draw()

            self.norm_groupBox.setVisible(True)
            self.pb_ApplyNrmValues.setVisible(False)
            self.pushButtonForPdf.setEnabled(True)
            self.pb_ViewPdf.setEnabled(True)

            self.lst_Analyze.setVisible(True)
            self.pbAppendButton.setVisible(True)
            self.pbRemoveButton.setVisible(True)

            filename=os.path.basename(self.audio_file_path)
            self.lst_Analyze.addItem(filename)

            self.pushButtonForPdf.clicked.connect(self.print_pdf)
            self.pb_ViewPdf.clicked.connect(self.view_pdf)

            self.canvas.draw()

    def cepstrum_individual(self):

        self.sr,self.data = read(self.individual_audio)
        
        if self.data is not None or self.sr == 0:
        
            n = len(self.data)
            normalize = n / 2
            fourier = fft(self.data)
            sampling_rate = self.sr
            
            log_spectrum = np.log(np.abs(fourier))
            cepstrum_result = np.fft.ifft(log_spectrum)
            
            self.canvas.figure.clear()
            self.canvas.figure.set_size_inches(self.default_canvas)
            self.ax = self.canvas.figure.add_subplot(111)

            self.ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            self.ax.yaxis.set_major_locator(plt.MaxNLocator(10))
            
            frequency_axis = fftfreq(n, d=1.0 / sampling_rate)
            norm_amplitude = np.abs(fourier) / normalize
            self.ax.plot(frequency_axis[:len(frequency_axis) // 2], norm_amplitude[:len(norm_amplitude) // 2], label='Spectrum')
            
            quefrency_axis = fftfreq(len(cepstrum_result), d=1.0 / sampling_rate)
            positive_part = np.abs(cepstrum_result)[:len(cepstrum_result) // 2]
            self.ax.plot(quefrency_axis[:len(quefrency_axis) // 2], positive_part, label='Cepstrum (Positive Part)')
            
            self.ax.set_xlabel('Frequency (Hz)')
            self.ax.set_ylabel('Amplitude')

            self.ax.yaxis.set_label_coords(-0.1, 0.5)

            self.ax.set_xlim(0, max(quefrency_axis))
            self.ax.set_ylim(0, max(norm_amplitude))

            self.ax.spines['right'].set_color('blue')
            self.ax.spines['top'].set_color('blue')
            self.ax.spines['bottom'].set_color('blue')
            self.ax.spines['left'].set_color('blue')
            self.ax.xaxis.label.set_color('blue')
            self.ax.yaxis.label.set_color('blue')
            self.ax.xaxis.label.set_color('blue')
            self.ax.tick_params(axis='x', colors='blue')
            self.ax.tick_params(axis='y', colors='blue')
            self.ax.patch.set_edgecolor('blue')
            
            self.canvas.draw()

            self.norm_groupBox.setVisible(True)
            self.pb_ApplyNrmValues.setVisible(False)
            self.pushButtonForPdf.setEnabled(True)
            self.pb_ViewPdf.setEnabled(True)

            self.pb_SaveRawData.setVisible(True)
            
    def processCentroid(self):

        self.sr,self.data = read(self.audio_file_path)

        self.grpStatistics.setEnabled(True)

        self.norm_groupBox.setVisible(False)
        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)
        self.lst_Analyze.setVisible(True)
        self.pbAppendButton.setVisible(True)
        self.pbRemoveButton.setVisible(True)

        filename=os.path.basename(self.audio_file_path)
        self.lst_Analyze.addItem(filename)


        if self.grpStatistics_mean.isChecked():
            self.spectralCentroid_statistics('mean')
        elif self.grpStatistics_std.isChecked():
            self.spectralCentroid_statistics('std')
        elif self.grpStatistics_min.isChecked():
            self.spectralCentroid_statistics('min')
        elif self.grpStatistics_max.isChecked():
            self.spectralCentroid_statistics('max')
        elif self.grpStatistics_median.isChecked():
            self.spectralCentroid_statistics('median')
        elif self.grpStatistics_maxbymean.isChecked():
            self.spectralCentroid_statistics('max_by_mean')
        elif self.grpStatistics_maxbymedian.isChecked():
            self.spectralCentroid_statistics('max_by_median')
        elif self.grpStatistics_stdbymean.isChecked():
            self.spectralCentroid_statistics('std_by_mean')

        self.pushButtonForPdf.clicked.connect(self.print_pdf)
        self.pb_ViewPdf.clicked.connect(self.view_pdf)

    def spectralCentroid_statistics(self, statistic):

        self.grpStatistics.setEnabled(True)
        self.canvas.figure.clear()

        self.canvas.figure.set_size_inches(self.default_canvas)

        ax = self.canvas.figure.add_subplot(111)

        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        self.sr == 22050
        
        f, t, Sxx = spectrogram(self.data, self.sr)
        centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)

        time_points = np.linspace(0, len(self.data) / self.sr, len(centroid))

        spectral_centroid = np.interp(f, np.linspace(f[0], f[-1], len(centroid)), centroid)
        

 

        if spectral_centroid is None:
            return

        value = None
        if statistic == 'mean':
            value = np.mean(spectral_centroid)
            print('spectral centroid mean:', value)
        elif statistic == 'min':
            value = np.min(spectral_centroid)
            print('spectral centroid min:', value)
        elif statistic == 'max':
            value = np.max(spectral_centroid)
            print('spectral centroid max:', value)
        elif statistic == 'median':
            value = np.median(spectral_centroid)
            print('spectral centroid median:', value)
        elif statistic == 'std':
            value = np.std(spectral_centroid)
            print('spectral centroid std:', value)
        elif statistic == 'max_by_mean':
            mean_val = np.mean(spectral_centroid)
            differences_mean = [mean_val - x for x in spectral_centroid]
            value = np.max(differences_mean)
            print('Max by Mean of Spectral Centroid:', value)
        elif statistic == 'max_by_median':
            median_val = np.median(spectral_centroid)
            differences_median = [median_val - x for x in spectral_centroid]
            value = np.max(differences_median)
            print('Max by Median of Spectral Centroid:', value)
        elif statistic == 'std_by_mean':
            mean_val = np.mean(spectral_centroid)
            differences_mean = [mean_val - x for x in spectral_centroid]
            value = np.std(differences_mean)
            print('Std by Mean of Spectral Centroid:', value)
        else:
            print("Invalid statistic.")
            return

        no_of_bins = self.le_NumOfBins.text()
        bin_number = int(no_of_bins)
 
        bins = np.linspace(min(f), max(f), bin_number)
        
        ax.hist(f, weights=np.sum(Sxx, axis=1), bins=bins, color='white', alpha=0.7, edgecolor='black')

        ax.set_ylabel('Occurrences')
        ax.yaxis.set_label_coords(-0.1, 0.5)

        plot_value = np.digitize(value, bins)
        ax.patches[plot_value].set_facecolor('yellow')

        ax.set_xlim(min(f), max(f))

        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')

        self.canvas.draw()

    def spectralCentroid_individual_initial(self):
        self.canvas.figure.clear()
        self.sr, self.data = read(self.individual_audio)
        ax = self.canvas.figure.add_subplot(111)
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        self.sr = 22050

        f, t, Sxx = spectrogram(self.data, self.sr)
        centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / np.sum(Sxx, axis=0)
        spectral_centroid = np.interp(f, np.linspace(f[0], f[-1], len(centroid)), centroid)
        
        ax.plot(0, 100, 'bo')
        
        ax.set_ylabel('Spectral Centroid')
        ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        ax.set_xlim(-1.5, np.max(spectral_centroid)-1)
        ax.get_xaxis().set_visible(False)

        ax.set_ylim(0, 200)

        self.canvas.draw()

        return ax


    def spectralCentroid_individual_append(self, statistic,x_lim):

        ax = self.canvas.figure.axes[0]

        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        f, t, Sxx = spectrogram(self.data, self.sr)
        power_spectrum = np.abs(Sxx) ** 2
        num_fft_bins = power_spectrum.shape[0]
        frequency_resolution = self.sr / num_fft_bins
        frequencies = np.arange(num_fft_bins) * frequency_resolution
        spectral_centroid = np.sum(frequencies[:, np.newaxis] * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0)
        
        if spectral_centroid is None:
            return

        if statistic == 'mean':
            value = np.mean(spectral_centroid)
            print('mean:', value)
        elif statistic == 'min':
            value = np.min(spectral_centroid)
            print('min:', value)
        elif statistic == 'max':
            value = np.max(spectral_centroid)
            print('max:', value)
        elif statistic == 'median':
            value = np.median(spectral_centroid)
            print('median:', value)
        elif statistic == 'std':
            value = np.std(spectral_centroid)
            print('std:', value)
        elif statistic == 'max_by_mean':
            mean_val = np.mean(spectral_centroid)
            differences_mean = [mean_val - x for x in spectral_centroid]
            value = np.max(differences_mean)
            print('Max by Mean of Spectral Centroid:', value)
        elif statistic == 'max_by_median':
            median_val = np.median(spectral_centroid)
            differences_median = [median_val - x for x in spectral_centroid]
            value = np.max(differences_median)
            print('Max by Median of Spectral Centroid:', value)
        elif statistic == 'std_by_mean':
            mean_val = np.mean(spectral_centroid)
            differences_mean = [mean_val - x for x in spectral_centroid]
            value = np.std(differences_mean)
            print('Std by Mean of Spectral Centroid:', value)
        else:
            print("Invalid statistic.")
            return
        if value > 200:
            y_value = 200
        elif value < 10:
            y_value = 10
        else:
            y_value = value 

        xdistance = 8
        
        last_x = ax.lines[-1].get_xdata()[-1] if ax.lines else 0

        ax.scatter(last_x + xdistance, y_value, color='blue')

        ax.set_ylabel('Spectral centroid')
        ax.yaxis.set_label_coords(-0.1, 0.5)
        

        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        ax.set_ylim(0, 200)

        self.canvas.draw()
    

    def processSpread(self):

        self.sr,self.data = read(self.audio_file_path)

        self.grpStatistics.setEnabled(True)

        self.norm_groupBox.setVisible(False)
        self.pushButtonForPdf.setEnabled(True)
        self.pb_ViewPdf.setEnabled(True)
        self.lst_Analyze.setVisible(True)
        self.pbAppendButton.setVisible(True)
        self.pbRemoveButton.setVisible(True)

        filename=os.path.basename(self.audio_file_path)
        self.lst_Analyze.addItem(filename)


        if self.grpStatistics_mean.isChecked():
            self.spectralSpread_statistics('mean')
        elif self.grpStatistics_std.isChecked():
            self.spectralSpread_statistics('std')
        elif self.grpStatistics_min.isChecked():
            self.spectralSpread_statistics('min')
        elif self.grpStatistics_max.isChecked():
            self.spectralSpread_statistics('max')
        elif self.grpStatistics_median.isChecked():
            self.spectralSpread_statistics('median')
        elif self.grpStatistics_maxbymean.isChecked():
            self.spectralSpread_statistics('max_by_mean')
        elif self.grpStatistics_maxbymedian.isChecked():
            self.spectralSpread_statistics('max_by_median')
        elif self.grpStatistics_stdbymean.isChecked():
            self.spectralSpread_statistics('std_by_mean')

        self.pushButtonForPdf.clicked.connect(self.print_pdf)
        self.pb_ViewPdf.clicked.connect(self.view_pdf)

    def spectralSpread_statistics(self, statistic):

        self.grpStatistics.setEnabled(True)
        self.canvas.figure.clear()
        
        self.canvas.figure.set_size_inches(self.default_canvas)
        ax = self.canvas.figure.add_subplot(111)

        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        self.sr == 22050

        f, t, Sxx = spectrogram(self.data, self.sr)
        power_spectrum = np.abs(Sxx) ** 2
        num_fft_bins = power_spectrum.shape[0]
        frequency_resolution = self.sr / num_fft_bins
        frequencies = np.arange(num_fft_bins) * frequency_resolution
        spectral_centroid = np.sum(frequencies[:, np.newaxis] * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0)
        spectral_spread = np.sqrt(np.sum(((frequencies[:, np.newaxis] - spectral_centroid) ** 2) * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0))

        if spectral_spread is None:
            return

        value = None
        if statistic == 'mean':
            value = np.mean(spectral_spread)
            print('spectral spread mean:', value)
        elif statistic == 'min':
            value = np.min(spectral_spread)
            print('spectral spread min:', value)
        elif statistic == 'max':
            value = np.max(spectral_spread)
            print('spectral spread max:', value)
        elif statistic == 'median':
            value = np.median(spectral_spread)
            print('spectral spread median:', value)
        elif statistic == 'std':
            value = np.std(spectral_spread)
            print('spectral spread std:', value)
        elif statistic == 'max_by_mean':
            mean_val = np.mean(spectral_spread)
            differences_mean = [mean_val - x for x in spectral_spread]
            value = np.max(differences_mean)
            print('Max by Mean of Spectral Spread:', value)
        elif statistic == 'max_by_median':
            median_val = np.median(spectral_spread)
            differences_median = [median_val - x for x in spectral_spread]
            value = np.max(differences_median)
            print('Max by Median of Spectral Spread:', value)
        elif statistic == 'std_by_mean':
            mean_val = np.mean(spectral_spread)
            differences_mean = [mean_val - x for x in spectral_spread]
            value = np.std(differences_mean)
            print('Std by Mean of Spectral Spread:', value)
        else:
            print("Invalid statistic.")
            return

        no_of_bins = self.le_NumOfBins.text()
        bin_number = int(no_of_bins)
        
        bins = np.linspace(min(frequencies), max(frequencies), bin_number)


        ax.hist(frequencies, weights=np.sum(Sxx, axis=1), bins=bins, color='white', alpha=0.7, edgecolor='black')

        ax.set_ylabel('Occurrences')
        ax.yaxis.set_label_coords(-0.1, 0.5)

        plot_value = np.digitize(value, bins)
        ax.patches[plot_value].set_facecolor('yellow')

        ax.set_xlim(min(frequencies), max(frequencies))

        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')

        self.canvas.draw()


    def spectralSpread_individual_initial(self):
        self.canvas.figure.clear()
        self.sr, self.data = read(self.individual_audio)
        ax = self.canvas.figure.add_subplot(111)
        ax.yaxis.set_major_locator(plt.MaxNLocator(10))

        self.sr ==22050

        f, t, Sxx = spectrogram(self.data, self.sr)
        power_spectrum = np.abs(Sxx) ** 2
        num_fft_bins = power_spectrum.shape[0]
        frequency_resolution = self.sr / num_fft_bins
        frequencies = np.arange(num_fft_bins) * frequency_resolution
        spectral_centroid = np.sum(frequencies[:, np.newaxis] * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0)
        print(spectral_centroid)
        spectral_spread = np.sqrt(np.sum(((frequencies[:, np.newaxis] - spectral_centroid) ** 2) * power_spectrum, axis=0) / np.sum(power_spectrum, axis=0))

        ax.plot(0, 100, 'bo')
        
        ax.set_ylabel('Spectral spread')
        ax.yaxis.set_label_coords(-0.1, 0.5)

        ax.spines['right'].set_color('blue')
        ax.spines['top'].set_color('blue')
        ax.spines['bottom'].set_color('blue')
        ax.spines['left'].set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.yaxis.label.set_color('blue')
        ax.xaxis.label.set_color('blue')
        ax.tick_params(axis='x', colors='blue')
        ax.tick_params(axis='y', colors='blue')
        ax.patch.set_edgecolor('blue')
        ax.set_xlim(-1.5, np.max(spectral_spread)-1)
        ax.get_xaxis().set_visible(False)

        ax.set_ylim(0, 200)

        self.canvas.draw()
            
    def clearWidget(self):
        self.canvas.figure.clear()
        self.canvas.draw()

    
    def disconnect_radio_button_signals(self):
        self.grpStatistics_mean.toggled.disconnect()
        self.grpStatistics_std.toggled.disconnect()
        self.grpStatistics_min.toggled.disconnect()
        self.grpStatistics_max.toggled.disconnect()
        self.grpStatistics_median.toggled.disconnect()


if __name__ == "__main__":

    app = QApplication(sys.argv)
    window = VoiceAnalyzer()
    window.setGeometry(100, 100, 1920, 1080)
    window.show()
    sys.exit(app.exec_())
