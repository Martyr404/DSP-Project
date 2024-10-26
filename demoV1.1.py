from PyQt5 import QtWidgets,QtGui,QtCore
import sys,time


import webrtcvad
import collections
import signal
import pyaudio
from array import array
from struct import pack
import wave
import pygame

import numpy as np
import soundfile as sf


import librosa
import matplotlib.pyplot as plt
import scipy.fftpack as fft
from scipy.signal import medfilt


class RecordingThread(QtCore.QThread):
    signal=QtCore.pyqtSignal(str)
    def __init__(self,output_file,parent=None,index=0):
        super(RecordingThread,self).__init__(parent)
        self.output_file=output_file
        self.index=index
        self.isRunning=True

    def run(self):
        print(f"Log: Start Running Recording Thread No.{self.index}\n")
        # 音频配置
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK_DURATION_MS = 30  # 每个音频块的持续时间 (ms)
        PADDING_DURATION_MS = 1500  # 填充时间 (ms)
        CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # 每个音频块包含的帧数
        CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
        NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
        NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
        NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2
        START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

        vad = webrtcvad.Vad(1)

        # 初始化 Pygame 用于播放音频
        pygame.mixer.pre_init(RATE, -16, CHANNELS, 2048)  # 设置 mixer 避免音频延迟
        pygame.mixer.init()
        pygame.init()

        # 初始化 PyAudio
        pa = pyaudio.PyAudio()
        stream = pa.open(format=FORMAT,
                         channels=CHANNELS,
                         rate=RATE,
                         input=True,
                         start=False,
                         frames_per_buffer=CHUNK_SIZE)

        got_a_sentence = False
        while self.isRunning:
            ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)
            triggered = False
            voiced_frames = []
            ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
            ring_buffer_index = 0
            ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
            ring_buffer_index_end = 0
            buffer_in = ''

            raw_data = array('h')
            index = 0
            start_point = 0
            StartTime = time.time()

            print("Log: * recording start * \n")
            stream.start_stream()

            while not got_a_sentence and self.isRunning:
                chunk = stream.read(CHUNK_SIZE)
                raw_data.extend(array('h', chunk))
                index += CHUNK_SIZE
                TimeUse = time.time() - StartTime

                active = vad.is_speech(chunk, RATE)

                if active:
                    sys.stdout.write("Log: * Recording: Voice Activity Detect! *\n")

                    self.signal.emit("您说话了")

                else:
                    sys.stdout.write("Log: * Recording:Sleep * \n")
                    self.signal.emit("闲置中")

                ring_buffer_flags[ring_buffer_index] = 1 if active else 0
                ring_buffer_index += 1
                ring_buffer_index %= NUM_WINDOW_CHUNKS

                ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
                ring_buffer_index_end += 1
                ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

                if not triggered:
                    ring_buffer.append(chunk)
                    num_voiced = sum(ring_buffer_flags)
                    if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                        sys.stdout.write('Log: * Open * \n')
                        triggered = True
                        start_point = index - CHUNK_SIZE * 20  # 起始点
                        ring_buffer.clear()
                else:
                    ring_buffer.append(chunk)
                    num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)

                    if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or TimeUse > 10:
                        sys.stdout.write('Log: * Close * \n')
                        triggered = False
                        got_a_sentence = True

                sys.stdout.flush()

            sys.stdout.write('\n')

            stream.stop_stream()
            print("Log:  * done recording *\n")
            got_a_sentence = False

            # 裁剪静音部分
            raw_data.reverse()
            for index in range(start_point):
                raw_data.pop()
            raw_data.reverse()

            # 归一化音频数据
            times = float(32767) / max(abs(i) for i in raw_data)
            r = array('h')
            for time_item in raw_data:
                r.append(int(time_item * times))
            raw_data=r

            # 保存音频到指定路径

            wf = wave.open(self.output_file, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pa.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(pack('<' + ('h' * len(raw_data)), *raw_data))
            wf.close()
            print(f"Log: Audio auto saved to {self.output_file}\n")

            # 播放音频
            sound = pygame.mixer.Sound(buffer=pack('<' + ('h' * len(raw_data)), *raw_data))
            sound.play()

            # 等待音频播放完成
            while pygame.mixer.get_busy():
                pass

        stream.close()
        pa.terminate()


    def stop(self):
        self.isRunning=False
        print(f"Log: Stop Running thread No.{self.index}\n")
        self.signal.emit("成功结束录音")
        try:
            self.signal.disconnect()
        except TypeError:
            pass
        time.sleep(2.5)

        self.terminate()

class EchoAppendingThread(QtCore.QThread):
    signal=QtCore.pyqtSignal(str)
    def __init__(self,input_path,output_path,index=3,parent=None):
        super(EchoAppendingThread,self).__init__(parent)
        self.index=index
        self.input_file=input_path
        self.output_file=output_path
        self.isRunning=True
    def run(self):
        print("Log: EchoAppending Action Executed\n")
        decay=0.5
        delay=0.5
        try:
            data, sample_rate = sf.read(self.input_file)
            delay_samples = int(delay * sample_rate)
            echo_data = np.zeros(len(data) + delay_samples)
            echo_data[:len(data)] = data
            # 将延迟后的信号添加到原始信号中，形成回声效果
            for i in range(len(data)):
                echo_data[i + delay_samples] += decay * data[i]

                # 将带有回声效果的音频数据写入输出文件
            sf.write(self.output_file, echo_data, sample_rate)
            print("Log: Echo Addition Action Successfully Executed\n")
            self.signal.emit("回声添加成功")
            self.deleteLater()
        except:
            print("Log: Incorrect File Path\n")
            self.signal.emit("ERROR\n输入输出文件路径有误")
            self.deleteLater()

    def stop(self):
        self.isRunning=False
        print("Log: EchoAppending Thread suspended\n")
        self.terminate()


class EchoEliminationThread(QtCore.QThread):
    signal = QtCore.pyqtSignal(str)

    def __init__(self, input_path, output_path, count=7,index=4, parent=None):
        super(EchoEliminationThread, self).__init__(parent)
        self.index = index
        self.input_file = input_path
        self.output_file = output_path
        self.count=count
        self.isRunning = True

    def run(self):
        print("Log: EchoElimination Action Executed\n")

        try:
            y, sr = librosa.load(self.input_file, sr=None)
            y_clean = y
            for i in range(0, self.count):
                s_full, phase = librosa.magphase(librosa.stft(y_clean))
                noise_power = np.mean(s_full[:, :int(sr * 0.1)], axis=1)
                mask = s_full > noise_power[:, None]
                mask = mask.astype(float)
                mask = medfilt(mask, kernel_size=(1, 5))
                s_clean = s_full * mask
                y_clean = librosa.istft(s_clean * phase)
            sf.write(self.output_file, y_clean, sr)
            print("Log: Echo Elimination Action Successfully Executed\n")
            self.signal.emit("回声消除成功")
            self.deleteLater()
        except:
            print("Log: Incorrect File Path\n")
            self.signal.emit("ERROR\n输入输出文件路径有误")
            self.deleteLater()

    def stop(self):
        self.isRunning = False
        print("Log: EchoElimination Thread suspended\n")
        self.terminate()

class VoiceAmplifiedThread(QtCore.QThread):
    signal=QtCore.pyqtSignal(str)
    def __init__(self,input_path,output_path,index=5,parent=None):
        super(VoiceAmplifiedThread,self).__init__(parent)
        self.index=index
        self.input_file=input_path
        self.output_file=output_path
        self.isRunning=True
    def run(self):
        print("Log: Voice Amplified Action Executed\n")
        gain=1.5
        try:
            # 读取音频文件
            data, sample_rate = sf.read(self.input_file)

            # 增加音量，将音频数据乘以增益因子
            amplified_data = data * gain

            # 防止音频剪切，将音频数据限制在 [-1.0, 1.0] 的范围内
            amplified_data = np.clip(amplified_data, -1.0, 1.0)

            # 将增强后的音频数据写入输出文件
            sf.write(self.output_file, amplified_data, sample_rate)
            print("Log: Voice Amplified Action Successfully Executed\n")
            self.signal.emit("人声增强成功")
            self.deleteLater()
        except:
            print("Log: Incorrect File Path\n")
            self.signal.emit("ERROR\n输入输出文件路径有误")
            self.deleteLater()

    def stop(self):
        self.isRunning=False
        print("Log: Voice Amplified Thread suspended\n")
        self.terminate()

class VoiceAttenuationThread(QtCore.QThread):
    signal=QtCore.pyqtSignal(str)
    def __init__(self,input_path,output_path,index=6,parent=None):
        super(VoiceAttenuationThread,self).__init__(parent)
        self.index=index
        self.input_file=input_path
        self.output_file=output_path
        self.isRunning=True
    def run(self):
        print("Log: Voice Amplified Action Executed\n")
        gain=0.5
        try:
            # 读取音频文件
            data, sample_rate = sf.read(self.input_file)

            # 增加音量，将音频数据乘以增益因子
            amplified_data = data * gain

            # 防止音频剪切，将音频数据限制在 [-1.0, 1.0] 的范围内
            amplified_data = np.clip(amplified_data, -1.0, 1.0)

            # 将增强后的音频数据写入输出文件
            sf.write(self.output_file, amplified_data, sample_rate)
            print("Log: Voice Attenuation Action Successfully Executed\n")
            self.signal.emit("人声减弱成功")
            self.deleteLater()
        except:
            print("Log: Incorrect File Path\n")
            self.signal.emit("ERROR\n输入输出文件路径有误")
            self.deleteLater()

    def stop(self):
        self.isRunning=False
        print("Log: Voice Attenuation Thread suspended\n")
        self.terminate()

class dspWindows(QtWidgets.QMainWindow):

    def __init__(self):
        super(dspWindows, self).__init__()
        self.setGeometry(10, 10,650,700)
        self.setFixedSize(650,700)
        self.setWindowTitle("DSP Project")
        self.setWindowIcon(QtGui.QIcon("icon.png"))
        self.move(400,500)
        self.initUI()

        #后台线程函数
        self.recordingThread={}
        self.echoAppendingThread={}
        self.echoEliminationThread={}
        self.voiceAmplifiedThread={}
        self.voiceAttenuationThread={}

        #按钮事件连接
        self.StartRecordingButton.clicked.connect(self.OnStartRecordingButtonClicked)
        self.StopRecordingButton.clicked.connect(self.OnStopRecordingButtonClicked)
        self.EchoAppendButton.clicked.connect(self.OnEchoAppendingButtonClicked)
        self.EchoEliminationButton.clicked.connect(self.OnEchoEliminationButtonClicked)
        self.VoiceAmplifiedButton.clicked.connect(self.OnVoiceAmplifedButtonClicked)
        self.VoiceAttenuationButton.clicked.connect(self.OnVoiceAttenuationButtonClicked)


    def OnStartRecordingButtonClicked(self):
        self.recordingThread[1]=RecordingThread(parent=None,output_file=self.outputTextBlock.text(),index=1)
        self.recordingThread[1].start()
        self.recordingThread[1].signal.connect(self.UpdateGUI)
        self.StartRecordingButton.setEnabled(False)
        self.StopRecordingButton.setEnabled(True)

    def OnStopRecordingButtonClicked(self):
        self.recordingThread[1].signal.connect(self.UpdateGUI)
        self.recordingThread[1].stop()
        self.StopRecordingButton.setEnabled(False)
        self.StartRecordingButton.setEnabled(True)

    def OnEchoAppendingButtonClicked(self):
        self.echoAppendingThread[1]=EchoAppendingThread(self.inputTextBlock.text(),self.outputTextBlock.text(),3,
                                                        None)
        self.echoAppendingThread[1].start()
        self.echoAppendingThread[1].signal.connect(self.UpdateGUI)
    def OnEchoEliminationButtonClicked(self):
        self.echoEliminationThread[1] = EchoEliminationThread(self.inputTextBlock.text(), self.outputTextBlock.text(), 7,
                                                          4,None)
        self.echoEliminationThread[1].start()
        self.echoEliminationThread[1].signal.connect(self.UpdateGUI)

    def OnVoiceAmplifedButtonClicked(self):
        self.voiceAmplifiedThread[1]=VoiceAmplifiedThread(self.inputTextBlock.text(),self.outputTextBlock.text(),5)
        self.voiceAmplifiedThread[1].start()
        self.voiceAmplifiedThread[1].signal.connect(self.UpdateGUI)


    def OnVoiceAttenuationButtonClicked(self):
        self.voiceAttenuationThread[1] = VoiceAttenuationThread(self.inputTextBlock.text(), self.outputTextBlock.text(), 6)
        self.voiceAttenuationThread[1].start()
        self.voiceAttenuationThread[1].signal.connect(self.UpdateGUI)

    #更新UI函数
    def UpdateGUI(self,message):
        msg=message
        index=self.sender().index
        isRecording=self.sender().isRunning
        if index==1 and isRecording==True:
            print(msg)
            self.CurrentStateLabel2.setText(msg)
        if index==1 and isRecording==False:
            print(msg)
            self.CurrentStateLabel2.setText(msg)
        if index==3:
            self.CurrentStateLabel2.setText(msg)
        if index==4:
            self.CurrentStateLabel2.setText(msg)
        if index==5:
            self.CurrentStateLabel2.setText(msg)
        if index==6:
            self.CurrentStateLabel2.setText(msg)



    def initUI(self):

        PathFont=QtGui.QFont("Arial",9)
        PathFont.setItalic(True)

        self.inputTextLabel = QtWidgets.QLabel(self)
        self.inputTextLabel.setText("Please input the targeted file path below:")
        self.inputTextLabel.move(40, 10)
        self.inputTextLabel.resize(450,100)
        self.inputTextLabel.setFont(QtGui.QFont("Arial",12))

        self.inputTextBlock =QtWidgets.QLineEdit(self)
        self.inputTextBlock.setText("Defalut Path")
        self.inputTextBlock.move(40,85)
        self.inputTextBlock.resize(250,25)
        self.inputTextBlock.setFont(PathFont)
        self.inputTextBlock.setStyleSheet("padding-left: 10px;")

        self.outputTextLabel = QtWidgets.QLabel(self)
        self.outputTextLabel.setText("Please input the output file path below:")
        self.outputTextLabel.move(40, 100)
        self.outputTextLabel.resize(450, 100)
        self.outputTextLabel.setFont(QtGui.QFont("Arial", 12))

        self.outputTextBlock = QtWidgets.QLineEdit(self)
        self.outputTextBlock.setText("Defalut Path")
        self.outputTextBlock.move(40, 175)
        self.outputTextBlock.resize(250, 25)
        self.outputTextBlock.setFont(PathFont)
        self.outputTextBlock.setStyleSheet("padding-left: 10px;")

        self.StartRecordingButton=QtWidgets.QPushButton(self)
        self.StartRecordingButton.setText("开始录制")
        self.StartRecordingButton.setFont(QtGui.QFont("SimHei",12))
        self.StartRecordingButton.resize(170,50)
        self.StartRecordingButton.move(40,250)

        self.EchoAppendButton = QtWidgets.QPushButton(self)
        self.EchoAppendButton.setText("回声添加")
        self.EchoAppendButton.setFont(QtGui.QFont("SimHei", 12))
        self.EchoAppendButton.resize(170, 50)
        self.EchoAppendButton.move(230, 250)


        self.StopRecordingButton = QtWidgets.QPushButton(self)
        self.StopRecordingButton.setText("停止录制")
        self.StopRecordingButton.setFont(QtGui.QFont("SimHei", 12))
        self.StopRecordingButton.resize(170,50)
        self.StopRecordingButton.move(40,320)
        self.StopRecordingButton.setEnabled(False)

        self.EchoEliminationButton = QtWidgets.QPushButton(self)
        self.EchoEliminationButton.setText("回声消除")
        self.EchoEliminationButton.setFont(QtGui.QFont("SimHei", 12))
        self.EchoEliminationButton.resize(170, 50)
        self.EchoEliminationButton.move(230, 320)

        self.VoiceAmplifiedButton = QtWidgets.QPushButton(self)
        self.VoiceAmplifiedButton.setText("人声加强")
        self.VoiceAmplifiedButton.setFont(QtGui.QFont("SimHei", 12))
        self.VoiceAmplifiedButton.resize(170, 50)
        self.VoiceAmplifiedButton.move(420, 250)


        self.VoiceAttenuationButton = QtWidgets.QPushButton(self)
        self.VoiceAttenuationButton.setText("人声减弱")
        self.VoiceAttenuationButton.setFont(QtGui.QFont("SimHei", 12))
        self.VoiceAttenuationButton.resize(170, 50)
        self.VoiceAttenuationButton.move(420, 320)


        self.CurrentStateLabel1 = QtWidgets.QLabel(self)
        self.CurrentStateLabel1.setText("程序当前状态:")
        self.CurrentStateLabel1.setFont(QtGui.QFont("SimHei", 16))
        self.CurrentStateLabel1.resize(200,40)
        self.CurrentStateLabel1.move(40,400)

        self.CurrentStateLabel2 = QtWidgets.QLabel(self)
        self.CurrentStateLabel2.setText("未运行")
        self.CurrentStateLabel2.setAlignment(QtCore.Qt.AlignCenter)
        self.CurrentStateLabel2.setStyleSheet("background-color: white;")
        self.CurrentStateLabel2.setFont(QtGui.QFont("SimHei", 12))
        self.CurrentStateLabel2.resize(360, 70)
        self.CurrentStateLabel2.move(40, 460)

        self.ImageLabel = QtWidgets.QLabel(self)
        self.ImageLabel.setGeometry(430, 460, 150, 150)
        self.ImageLabel.setPixmap(QtGui.QPixmap("what2.jpg"))
        self.ImageLabel.adjustSize()



def windows():
    app = QtWidgets.QApplication(sys.argv)
    win = dspWindows()
    win.show()
    sys.exit(app.exec())

windows()
#1449