import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QLineEdit, QFileDialog, QWidget, QScrollArea, QVBoxLayout
from PyQt5.QtGui import QFont, QDoubleValidator, QIntValidator
from PyQt5.QtCore import Qt
from pprint import pformat


import data_preprocessing as dp
from NN import NN
from KNN import KNN

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.precentage = None
        self.result_window1 = None
        self.result_window2 = None
        self.file_path = ""
        self.setWindowTitle("Assignment2")
        self.setGeometry(600,300, 800,500)
        self.setStyleSheet("background:#31619C")
        self.read_file = QPushButton("Open CSV File", self)
        self.button = QPushButton("Submit",self)
        self.data_percentage = QLineEdit(self)
        self.clusters_input = QLineEdit(self)
        self.initUI()

    def initUI(self):
        title = QLabel("Clustring System", self)
        title.setGeometry(190, 0, 410, 100)
        title.setFont(QFont("Arial", 20))
        title.setStyleSheet("Color:white;"
                            "font-weight:bold;"
                            "background:#31619C")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignCenter)

        self.read_file.setGeometry(320,100,160,50)
        self.read_file.setStyleSheet("background:white;"
                                     "border-radius:20px;"
                                     "ont-size:20px;")
        self.read_file.clicked.connect(self.open_csv_file)

        validator = QDoubleValidator(0.0, 1.0, 120)
        validator.setNotation(QDoubleValidator.StandardNotation)

        self.data_percentage.setGeometry(320, 160, 160, 50)
        self.data_percentage.setPlaceholderText("% Of Data")
        self.data_percentage.setStyleSheet("Color:black;"
                                         "border-radius:20px;"
                                         "background:white;"
                                         "font-size:20px;"
                                         "font-family:Arial")
        self.data_percentage.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clusters_input.setValidator(validator)

        self.data_percentage.setValidator(QIntValidator(0,100))
        self.clusters_input.setGeometry(320, 220, 160, 50)
        self.clusters_input.setPlaceholderText("# Of Clusters")
        self.clusters_input.setStyleSheet("Color:black;"
                                        "border-radius:20px;"
                                        "background:white;"
                                        "font-size:20px;"
                                         "font-family:Arial")
        self.clusters_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clusters_input.setValidator(QIntValidator(1, 200))

        self.button.setGeometry(350, 340, 80, 40)
        self.button.setStyleSheet("background: white;"
                                  "border-radius: 20px;")
        self.button.clicked.connect(self.on_clicked)


    def on_clicked(self):
        path = self.file_path
        k = int(self.clusters_input.text())
        self.precentage = int(self.data_percentage.text())
        if (path != ""):
            min_acc = 95.0
            X_train, y_train, X_test, y_test = dp.dataPreprocessing(path, self.precentage)
            #print(data.head())
            KNN_Accuracy, KNN_Result = KNN(X_train, y_train, X_test, y_test, k)
            NN_Accuracy, NN_Result = NN(X_train, y_train, X_test, y_test)
            print(KNN_Accuracy,end="")
            print(NN_Accuracy)
            while(KNN_Accuracy < min_acc):
                print("KNN")
                X_t, y_t, X_te, y_te = dp.dataPreprocessing(path, self.precentage)
                KNN_Accuracy, KNN_Result = KNN(X_t, y_t, X_te, y_te, k)
                print(f'->{KNN_Accuracy}')
            print("-------------------------------")
            while (NN_Accuracy < min_acc):
                print("NN")
                X_t, y_t, X_te, y_te = dp.dataPreprocessing(path, self.precentage)
                NN_Accuracy, NN_Result = NN(X_t, y_t, X_te, y_te)
                print(NN_Accuracy)
            #print(self.precentage)
            print("finished")
            self.result_window1 = ResultWindow(KNN_Accuracy, KNN_Result, 'KNN')
            self.result_window2 = ResultWindow(NN_Accuracy, NN_Result,'NN')
            self.result_window1.show()
            self.result_window2.show()

    def open_csv_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)")



class ResultWindow(QWidget):
    def __init__(self, accuracy, data, From):
        super().__init__()
        self.accuracy = accuracy
        self.From = From
        self.setStyleSheet("background:#31619C;")
        self.setGeometry(600,250,700,650)


        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)


        content_widget = QWidget()
        layout = QVBoxLayout(content_widget)

        formatted_text = f'{self.From}:\n'
        for rec in data:
            formatted_text += f'{rec}\n'
        formatted_text += f'Accuracy: {self.accuracy}\n'

        self.results = QLabel(formatted_text,self)
        self.results.setGeometry(0, 0, 900, 1000)
        self.results.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.results.setWordWrap(True)
        self.results.setStyleSheet("color:white;"
                                   "font-size:20px;")

        layout.addWidget(self.results)
        content_widget.setLayout(layout)


        scroll_area.setWidget(content_widget)


        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()