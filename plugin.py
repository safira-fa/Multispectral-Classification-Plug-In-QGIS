# -*- coding: utf-8 -*-
"""
Главный класс плагина + фоновый worker.
"""
import os
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox, QApplication
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog
from .dialog import MultispectralClassifierDialog
from .classifiers import SklearnClassifier, ClassificationUtils


# ==================== ФОНОВЫЙ WORKER ====================
class ClassificationWorker(QThread):
    """Выносит тяжёлую работу из GUI-потока."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, input_path, model_path):
        super().__init__()
        self.input_path = input_path
        self.model_path = model_path
        self._is_canceled = False

    def cancel(self):
        self._is_canceled = True

    def run(self):
        try:
            base_name = os.path.splitext(os.path.basename(self.input_path))[0]
            out_dir = os.path.dirname(self.input_path)
            out_path = os.path.join(out_dir, f'{base_name}_ml_classified.tif')

            def cb(percent, msg):
                if self._is_canceled:
                    raise InterruptedError("Отменено пользователем")
                self.progress.emit(percent, msg)

            self.progress.emit(5, "Загрузка модели...")
            classifier = SklearnClassifier(self.model_path)

            self.progress.emit(10, "Предсказание...")
            out_path = classifier.predict(
                raster_path=self.input_path,
                output_path=out_path,
                block_size=1024,
                progress_callback=cb
            )

            self.progress.emit(95, "Сохранение результата...")
            self.finished.emit(out_path)

        except InterruptedError:
            self.error.emit("Классификация отменена.")
        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), 'MultispectralClassifier')
            self.error.emit(str(e))


# ==================== ГЛАВНЫЙ КЛАСС ПЛАГИНА ====================
class MultispectralClassifierPlugin:

    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = 'Классификация снимков'
        self.toolbar = self.iface.addToolBar('MultispectralClassifier')
        self.toolbar.setObjectName('MultispectralClassifier')
        self.dialog = None
        self.worker = None

    def initGui(self):
        icon_path = os.path.join(self.plugin_dir, 'icon.png')
        self.action = QAction(
            QIcon(icon_path) if os.path.exists(icon_path) else QIcon(),
            'ML-классификация мультиспектрального снимка',
            self.iface.mainWindow()
        )
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu(self.menu, self.action)
        self.iface.addToolBarIcon(self.action)
        self.actions.append(self.action)

    def unload(self):
        for act in self.actions:
            self.iface.removePluginMenu(self.menu, act)
            self.iface.removeToolBarIcon(act)
        del self.toolbar

    def run(self):
        self.dialog = MultispectralClassifierDialog(
            self.iface, self.iface.mainWindow()
        )
        # Добавляем кнопку разворачивания на весь экран
        flags = self.dialog.windowFlags()
        flags |= Qt.WindowMaximizeButtonHint  # Кнопка максимизации
        flags |= Qt.WindowMinMaxButtonsHint   # Кнопки минимизации и максимизации
        self.dialog.setWindowFlags(flags)
    
        self.dialog.run_requested.connect(self.start_classification)
        self.dialog.show()

    def start_classification(self):
        params = self.dialog.get_parameters()

        if not params['input_path'] or not os.path.exists(params['input_path']):
            QMessageBox.warning(self.dialog, 'Ошибка', 'Выберите входной растр')
            return
        if not params.get('model_path'):
            QMessageBox.warning(self.dialog, 'Ошибка', 'Модель не выбрана')
            return

        self.dialog.progress_bar.setVisible(True)
        self.dialog.progress_bar.setValue(0)
        self.dialog.run_btn.setEnabled(False)
        self.dialog.cancel_btn.setText('Отменить')

        self.worker = ClassificationWorker(
            input_path=params['input_path'],
            model_path=params['model_path'],
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.dialog.cancel_btn.clicked.connect(self.worker.cancel)
        self.worker.start()

    def _on_progress(self, percent, msg):
        self.dialog.progress_bar.setValue(percent)
        self.dialog.progress_bar.setFormat(f'{percent}% - {msg}')
        QApplication.processEvents()

    def _on_finished(self, output_path):
        self.dialog.progress_bar.setVisible(False)
        self.dialog.run_btn.setEnabled(True)
        self.dialog.cancel_btn.setText('Отмена')

        if output_path and os.path.exists(output_path):
            layer_name = os.path.splitext(os.path.basename(output_path))[0]
            layer = QgsRasterLayer(output_path, layer_name)

            if layer.isValid():
                ClassificationUtils.apply_style(layer)
                QgsProject.instance().addMapLayer(layer)
                self.iface.messageBar().pushSuccess(
                    'Готово', f'Слой "{layer_name}" добавлен.'
                )

                self.dialog.last_output_path = output_path

                report_text = ClassificationUtils.generate_report_text(
                    classification_result_path=output_path,
                    source_input_path=self.dialog.get_parameters()['input_path']
                )
                self.dialog.set_report_text(report_text)

                self.dialog.tab_widget.setCurrentIndex(1)
            else:
                QMessageBox.critical(
                    self.dialog, 'Ошибка', f'Не удалось загрузить: {output_path}'
                )
        else:
            QMessageBox.critical(self.dialog, 'Ошибка', 'Файл результата не создан')

    def _on_error(self, msg):
        self.dialog.progress_bar.setVisible(False)
        self.dialog.run_btn.setEnabled(True)
        self.dialog.cancel_btn.setText('Отмена')
        QMessageBox.critical(self.dialog, 'Ошибка классификации', msg)