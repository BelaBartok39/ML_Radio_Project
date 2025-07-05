#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: ML_Radio_Project
# Author: Nick Redmond
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import analog
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
import modulation_inference_flowgraph_epy_block_0 as epy_block_0  # embedded python block
import threading



class modulation_inference_flowgraph(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "ML_Radio_Project", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("ML_Radio_Project")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "modulation_inference_flowgraph")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 1e6
        self.samp_per_sym = samp_per_sym = 4
        self.prediction = prediction = 'N/A'
        self.mod_type = mod_type = 0
        self.excess_bw = excess_bw = 0.35

        ##################################################
        # Blocks
        ##################################################

        self._prediction_tool_bar = Qt.QToolBar(self)

        if None:
            self._prediction_formatter = None
        else:
            self._prediction_formatter = lambda x: str(x)

        self._prediction_tool_bar.addWidget(Qt.QLabel("Prediction:"))
        self._prediction_label = Qt.QLabel(str(self._prediction_formatter(self.prediction)))
        self._prediction_tool_bar.addWidget(self._prediction_label)
        self.top_layout.addWidget(self._prediction_tool_bar)
        self.epy_block_0 = epy_block_0.blk()
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=digital.constellation_bpsk().base(),
            differential=True,
            samples_per_symbol=samp_per_sym,
            pre_diff_code=True,
            excess_bw=excess_bw,
            verbose=True,
            log=True,
            truncate=False)
        self.analog_sig_source_x_0 = analog.sig_source_b(samp_rate, analog.GR_SIN_WAVE, 100000, 1, 0, 0)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.epy_block_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "modulation_inference_flowgraph")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.analog_sig_source_x_0.set_sampling_freq(self.samp_rate)

    def get_samp_per_sym(self):
        return self.samp_per_sym

    def set_samp_per_sym(self, samp_per_sym):
        self.samp_per_sym = samp_per_sym

    def get_prediction(self):
        return self.prediction

    def set_prediction(self, prediction):
        self.prediction = prediction
        Qt.QMetaObject.invokeMethod(self._prediction_label, "setText", Qt.Q_ARG("QString", str(self._prediction_formatter(self.prediction))))

    def get_mod_type(self):
        return self.mod_type

    def set_mod_type(self, mod_type):
        self.mod_type = mod_type

    def get_excess_bw(self):
        return self.excess_bw

    def set_excess_bw(self, excess_bw):
        self.excess_bw = excess_bw




def main(top_block_cls=modulation_inference_flowgraph, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
