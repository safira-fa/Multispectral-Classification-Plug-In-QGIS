# -*- coding: utf-8 -*-
"""
Диалоговое окно ML-классификации мультиспектральных снимков.
Компактная версия с пастельно-синим стилем.
"""
import os
import shutil
import tempfile
import numpy as np
from qgis.PyQt import uic
from qgis.PyQt.QtWidgets import (
    QDialog, QFileDialog, QMessageBox,
    QInputDialog, QListWidgetItem,
    QVBoxLayout, QPushButton, QTextEdit
)
from qgis.PyQt.QtCore import Qt, QSettings, pyqtSignal
from qgis.core import (
    QgsRasterLayer, QgsMessageLog, Qgis, QgsMapLayerProxyModel,
    QgsProject, QgsRectangle, QgsVectorFileWriter,
    QgsCoordinateTransformContext, QgsCoordinateReferenceSystem,
    QgsCoordinateTransform
)
from osgeo import gdal

try:
    from .classifiers import SklearnClassifier
except ImportError:
    SklearnClassifier = None
    QgsMessageLog.logMessage("Warning: classifiers не найден.", "MultispectralPlugin")

try:
    from . import db_storage
except ImportError:
    db_storage = None

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'dialog_base.ui'))


# ==================== КОМПАКТНЫЙ СТИЛЬ ====================
BLUE_PASTEL_STYLE = """
QDialog {
    background-color: #EAF2FB;
    color: #2C3E50;
}

QTabWidget::pane {
    border: 1px solid #B8D4EC;
    background: #F5F9FD;
    padding: 4px;
}

QTabBar::tab {
    background: #D6E6F5;
    color: #2C3E50;
    padding: 8px 20px;           
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: bold;
    min-width: 120px;            
}

QTabBar::tab:selected {
    background: #4A90D9;
    color: white;
}

QGroupBox {
    background-color: #FFFFFF;
    border: 1px solid #B8D4EC;
    border-radius: 5px;
    margin-top: 8px;
    padding: 8px 6px 6px 6px;
    font-weight: bold;
    color: #2C5282;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    background-color: #F5F9FD;
}

QLineEdit, QComboBox, QListWidget, QTextEdit {
    border: 1px solid #B8D4EC;
    border-radius: 3px;
    padding: 3px 5px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #4A90D9;
}

QPushButton {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #6FA8DC, stop:1 #4A90D9);
    color: white;
    border: 1px solid #3A78B8;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #5A9FE3;
}
QPushButton:disabled {
    background-color: #C5D8EC;
    color: #7A8FA3;
}

QPushButton#run_btn {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #7BC47F, stop:1 #4CAF50);
}

QPushButton#btn_train_ml {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #FFB870, stop:1 #FF9800);
    border: 1px solid #E07D00;
}

QPushButton#btn_save_model_to_db {
    background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                      stop:0 #7BC47F, stop:1 #4CAF50);
    border: 1px solid #388E3C;
}

QProgressBar {
    border: 1px solid #B8D4EC;
    border-radius: 3px;
    text-align: center;
    background-color: #FFFFFF;
    height: 18px;
}
QProgressBar::chunk {
    background-color: #4A90D9;
}
"""


class MultispectralClassifierDialog(QDialog, FORM_CLASS):
    """Главный диалог плагина."""

    run_requested = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.current_ml_model_path = None
        self.last_report_text = ""
        self.last_output_path = ""
        self._is_loading_model = False  # ← ДОБАВИТЬ ФЛАГ
        self.setupUi(self)
        self.setMinimumSize(900, 750)
        self._apply_style()
        self._setup_extra_ui()
        self._setup_connections()

    def _apply_style(self):
        self.setStyleSheet(BLUE_PASTEL_STYLE)

    def _setup_extra_ui(self):
        self.raster_layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        self.labels_layer_combo.setFilters(QgsMapLayerProxyModel.VectorLayer)
        self.svm_params_widget.setVisible(False)
        self._toggle_raster_source()
        self._toggle_model_source()
        self._toggle_model_type()

    def _setup_connections(self):
        self.raster_from_project_radio.toggled.connect(self._toggle_raster_source)
        self.raster_from_file_radio.toggled.connect(self._toggle_raster_source)
        self.browse_tiff_btn.clicked.connect(self.browse_tiff_file)
        self.info_btn.clicked.connect(self.show_image_info)
        self.raster_layer_combo.layerChanged.connect(self._on_raster_layer_changed)

        self.train_new_model_radio.toggled.connect(self._toggle_model_source)
        self.load_pretrained_radio.toggled.connect(self._toggle_model_source)
        self.browse_pretrained_btn.clicked.connect(self.browse_pretrained_model)
        self.btn_train_ml.clicked.connect(self.on_ml_train_clicked)
        self.labels_layer_combo.layerChanged.connect(self.labels_field_combo.setLayer)

        self.model_type_combo.currentIndexChanged.connect(self._toggle_model_type)

        self.cancel_btn.clicked.connect(self.reject)
        self.run_btn.clicked.connect(self.on_run_clicked)

        if db_storage:
            self.btn_db_test.clicked.connect(self.test_db_connection)
            self.btn_db_refresh.clicked.connect(self.refresh_models_list)
            self.btn_db_load.clicked.connect(self.on_btn_db_load_clicked)
            self.models_list_widget.itemDoubleClicked.connect(self.load_model_from_db)

        self.btn_save_model_to_db.clicked.connect(self.save_current_model_to_db)
        self.btn_save_model_to_file.clicked.connect(self.save_model_to_file)

    def _toggle_raster_source(self):
        from_project = self.raster_from_project_radio.isChecked()
        self.raster_layer_combo.setVisible(from_project)
        self.tiff_path_edit.setVisible(not from_project)
        self.browse_tiff_btn.setVisible(not from_project)
    
        # ВАЖНО: добавляем setEnabled для активации/деактивации
        self.raster_layer_combo.setEnabled(from_project)
        self.tiff_path_edit.setEnabled(not from_project)
        self.browse_tiff_btn.setEnabled(not from_project)  # ← ЭТО ИСПРАВЛЯЕТ КНОПКУ

        if from_project:
            self._on_raster_layer_changed(self.raster_layer_combo.currentLayer())
        else:
            path = self.tiff_path_edit.text()
            if path:
                self.update_channels_info(path)

    def _toggle_model_source(self):
        train_new = self.train_new_model_radio.isChecked()
        self.train_widget.setVisible(train_new)
        self.pretrained_widget.setVisible(not train_new)
        self.pretrained_path_edit.setEnabled(not train_new)
        self.browse_pretrained_btn.setEnabled(not train_new)

    def _toggle_model_type(self):
        model_type = self.model_type_combo.currentText()
        if 'Random Forest' in model_type:
            self.rf_params_widget.setVisible(True)
            self.svm_params_widget.setVisible(False)
        else:
            self.rf_params_widget.setVisible(False)
            self.svm_params_widget.setVisible(True)

    def _on_raster_layer_changed(self, layer):
        if layer and self.raster_from_project_radio.isChecked():
            path = layer.source()
            if '|' in path:
                path = path.split('|')[0]
            self.update_channels_info(path)

    def update_channels_info(self, path):
        try:
            layer = QgsRasterLayer(path, "temp_layer")
            if not layer.isValid():
                raise Exception("Не удалось открыть растр")
            n_bands = layer.dataProvider().bandCount()
            width = layer.width()
            height = layer.height()
            crs = layer.crs().authid()
            self.channels_info_label.setText(
                f'Каналов: {n_bands} | Размер: {width}x{height} | CRS: {crs}'
            )
            self.channels_info_label.setStyleSheet('color: #2C5282; font-weight: bold;')
            self.channels_info_label.setVisible(True)
        except Exception as e:
            self.channels_info_label.setText(f'Ошибка: {str(e)}')
            self.channels_info_label.setStyleSheet('color: #c62828;')
            self.channels_info_label.setVisible(True)

    def browse_tiff_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите снимок', '', 'GeoTIFF (*.tif *.tiff);;Все файлы (*.*)'
        )
        if path:
            self.tiff_path_edit.setText(path)
            self.update_channels_info(path)

    def browse_pretrained_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Выберите предобученную модель', '', 'Joblib (*.joblib);;Все файлы (*.*)'
        )
        if path:
            self.pretrained_path_edit.setText(path)
            self.current_ml_model_path = path
            self.ml_status_label.setText(f'✓ Загружена: {os.path.basename(path)}')
            self.ml_status_label.setStyleSheet('color: #2C5282; font-style: italic;')

    def show_image_info(self):
        path = self._get_raster_path()
        if not path or not os.path.exists(path):
            self.iface.messageBar().pushWarning('Предупреждение', 'Сначала выберите растр')
            return

        try:
            layer = QgsRasterLayer(path, "info_layer")
            if not layer.isValid():
                raise Exception("Невалидный растр")

            p = layer.dataProvider()
            txt = (
                f"Файл: {os.path.basename(path)}\n"
                f"Размер: {layer.width()} x {layer.height()} пикселей\n"
                f"CRS: {layer.crs().authid()}\n"
                f"Каналов: {p.bandCount()}\n"
                f"Границы: {layer.extent().toString(3)}\n"
                f"Информация по каналам:\n"
            )

            for i in range(1, p.bandCount() + 1):
                dtype = p.dataType(i)
                nodata = p.sourceNoDataValue(i)
                type_name = {
                    Qgis.Byte: 'Byte', Qgis.UInt16: 'UInt16', Qgis.Int16: 'Int16',
                    Qgis.UInt32: 'UInt32', Qgis.Int32: 'Int32',
                    Qgis.Float32: 'Float32', Qgis.Float64: 'Float64',
                }.get(dtype, str(dtype))

                sample_w = min(64, layer.width())
                sample_h = min(64, layer.height())
                ext = layer.extent()
                center_x = (ext.xMinimum() + ext.xMaximum()) / 2
                center_y = (ext.yMinimum() + ext.yMaximum()) / 2
                sample_ext = QgsRectangle(
                    center_x - sample_w * layer.rasterUnitsPerPixelX() / 2,
                    center_y - sample_h * layer.rasterUnitsPerPixelY() / 2,
                    center_x + sample_w * layer.rasterUnitsPerPixelX() / 2,
                    center_y + sample_h * layer.rasterUnitsPerPixelY() / 2,
                )

                try:
                    block = p.block(i, sample_ext, sample_w, sample_h)
                    src_type = p.dataType(i)
                    np_dtype = np.uint16 if src_type in (Qgis.Byte, Qgis.UInt16, Qgis.Int16) else np.float32
                    arr = np.frombuffer(block.data(), dtype=np_dtype).astype(np.float32)
                    if nodata is not None and not np.isnan(nodata):
                        arr = arr[arr != nodata]
                    if arr.size > 0:
                        txt += (f"  Канал {i} [{type_name}]: "
                                f"min={arr.min():.2f}, max={arr.max():.2f}, mean={arr.mean():.2f}\n")
                except Exception:
                    pass

            dlg = QDialog(self)
            dlg.setWindowTitle('Информация о снимке')
            dlg.setMinimumSize(500, 400)
            lay = QVBoxLayout(dlg)
            te = QTextEdit()
            te.setPlainText(txt)
            te.setReadOnly(True)
            lay.addWidget(te)
            close = QPushButton('Закрыть')
            close.clicked.connect(dlg.accept)
            lay.addWidget(close)
            dlg.exec_()

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), "MultispectralPlugin", Qgis.Critical)
            self.iface.messageBar().pushCritical('Ошибка', str(e))

    def _get_raster_path(self):
        if self.raster_from_project_radio.isChecked():
            layer = self.raster_layer_combo.currentLayer()
            if layer:
                path = layer.source()
                if '|' in path:
                    path = path.split('|')[0]
                return path
            return ""
        else:
            return self.tiff_path_edit.text()

    def _get_raster_extent_wkt_4326(self):
        path = self._get_raster_path()
        if not path or not os.path.exists(path):
            return None
        try:
            layer = QgsRasterLayer(path, "tmp")
            if not layer.isValid():
                return None
            ext = layer.extent()
            crs_src = layer.crs()
            crs_dst = QgsCoordinateReferenceSystem("EPSG:4326")
            transform = QgsCoordinateTransform(crs_src, crs_dst, QgsProject.instance())
            ext_4326 = transform.transform(ext)
            return (f"POLYGON(({ext_4326.xMinimum()} {ext_4326.yMinimum()}, "
                    f"{ext_4326.xMaximum()} {ext_4326.yMinimum()}, "
                    f"{ext_4326.xMaximum()} {ext_4326.yMaximum()}, "
                    f"{ext_4326.xMinimum()} {ext_4326.yMaximum()}, "
                    f"{ext_4326.xMinimum()} {ext_4326.yMinimum()}))")
        except Exception as e:
            QgsMessageLog.logMessage(f"Не удалось получить WKT: {e}", 'MultispectralPlugin')
            return None

    def get_parameters(self):
        input_path = self._get_raster_path()
        output_dir = os.path.dirname(input_path) if input_path else ""

        model_path = None
        if self.train_new_model_radio.isChecked():
            model_path = self.current_ml_model_path
        else:
            model_path = self.pretrained_path_edit.text()

        model_type = 'rf' if 'Random Forest' in self.model_type_combo.currentText() else 'svm'

        ml_params = {'model_type': model_type}
        if model_type == 'rf':
            ml_params['n_estimators'] = int(self.rf_n_estimators.currentText())
            ml_params['max_depth'] = (
                None if self.rf_max_depth.currentText() == 'None'
                else int(self.rf_max_depth.currentText())
            )
        else:
            ml_params['svm_kernel'] = self.svm_kernel.currentText()
            ml_params['svm_C'] = float(self.svm_C.currentText())

        return {
            'input_path': input_path,
            'output_dir': output_dir,
            'model_path': model_path,
            'labels_layer_id': (
                self.labels_layer_combo.currentLayer().id()
                if self.train_new_model_radio.isChecked() and self.labels_layer_combo.currentLayer()
                else None
            ),
            'labels_field_name': (
                self.labels_field_combo.currentField()
                if self.train_new_model_radio.isChecked() else None
            ),
            'ml_params': ml_params
        }

    def on_ml_train_clicked(self):
        input_path = self._get_raster_path()
        if not input_path:
            QMessageBox.warning(self, "Ошибка", "Выберите входной растр!")
            return

        layer = self.labels_layer_combo.currentLayer()
        field = self.labels_field_combo.currentField()

        if layer is None:
            QMessageBox.warning(self, "Ошибка", "Векторный слой с метками не выбран!")
            return
        if not field:
            QMessageBox.warning(self, "Ошибка", "Поле с классами не определено!")
            return
        if layer.featureCount() == 0:
            QMessageBox.warning(self, "Ошибка", f"Слой '{layer.name()}' пуст!")
            return

        self._set_busy_state(True, "Растеризация и обучение...")

        try:
            self.ml_status_label.setText('Растеризация...')
            self.ml_status_label.setStyleSheet('color: #1565c0; font-style: italic;')

            ref_raster = QgsRasterLayer(input_path, "ref")
            if not ref_raster.isValid():
                raise RuntimeError("Не удалось открыть растр")

            temp_vec = tempfile.mktemp(suffix='.gpkg')
            ref_crs = ref_raster.crs()
            transform_context = QgsCoordinateTransformContext()
            options = QgsVectorFileWriter.SaveVectorOptions()
            options.driverName = 'GPKG'
            options.layerName = 'temp_layer'

            writer = QgsVectorFileWriter.create(
                temp_vec, layer.fields(), layer.wkbType(), ref_crs, transform_context, options
            )
            if writer.hasError():
                raise RuntimeError(f"Ошибка: {writer.errorMessage()}")

            for feature in layer.getFeatures():
                writer.addFeature(feature)
            del writer

            ref_ds = gdal.Open(input_path)
            geotransform = ref_ds.GetGeoTransform()
            projection = ref_ds.GetProjection()
            width = ref_ds.RasterXSize
            height = ref_ds.RasterYSize
            ref_ds = None

            temp_labels_tif = os.path.join(os.path.dirname(input_path), "temp_labels.tif")
            if os.path.exists(temp_labels_tif):
                try:
                    os.remove(temp_labels_tif)
                except Exception:
                    pass

            vec_ds = gdal.OpenEx(temp_vec, gdal.OF_VECTOR)
            vec_layer = vec_ds.GetLayerByName('temp_layer')

            driver = gdal.GetDriverByName('GTiff')
            out_ds = driver.Create(temp_labels_tif, width, height, 1, gdal.GDT_Byte, ['COMPRESS=LZW'])
            out_ds.SetGeoTransform(geotransform)
            out_ds.SetProjection(projection)
            out_band = out_ds.GetRasterBand(1)
            out_band.Fill(0)
            out_band.SetNoDataValue(0)

            err = gdal.RasterizeLayer(out_ds, [1], vec_layer, options=[f"ATTRIBUTE={field}"])
            if err != 0:
                raise RuntimeError(f"Ошибка растеризации: {err}")

            out_ds.FlushCache()
            arr = out_band.ReadAsArray()
            nonzero = np.count_nonzero(arr)
            out_ds = None
            vec_ds = None

            if nonzero == 0:
                raise RuntimeError("Полигоны не попали в область растра.")

            model_type = 'rf' if 'Random Forest' in self.model_type_combo.currentText() else 'svm'
            self.ml_status_label.setText(f'Обучение {model_type.upper()}...')

            classifier = SklearnClassifier()

            if model_type == 'rf':
                model_path = classifier.train(
                    raster_path=input_path,
                    labels_path=temp_labels_tif,
                    model_type='rf',
                    n_estimators=int(self.rf_n_estimators.currentText()),
                    max_depth=None if self.rf_max_depth.currentText() == 'None' else int(self.rf_max_depth.currentText())
                )
            else:
                model_path = classifier.train(
                    raster_path=input_path,
                    labels_path=temp_labels_tif,
                    model_type='svm',
                    svm_kernel=self.svm_kernel.currentText(),
                    svm_C=float(self.svm_C.currentText())
                )

            self.current_ml_model_path = model_path
            self.ml_status_label.setText(f'✓ Обучена: {os.path.basename(model_path)}')
            self.ml_status_label.setStyleSheet('color: #2C5282; font-style: italic;')

            for f in [temp_labels_tif, temp_vec]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

        except Exception as e:
            import traceback
            QgsMessageLog.logMessage(traceback.format_exc(), "MultispectralPlugin", Qgis.Critical)
            QMessageBox.critical(self, "Ошибка обучения", str(e))
            self.ml_status_label.setText('✗ Ошибка')
            self.ml_status_label.setStyleSheet('color: #c62828; font-style: italic;')
        finally:
            self._set_busy_state(False)

    def _set_busy_state(self, busy, message=""):
        self.browse_tiff_btn.setEnabled(not busy and self.raster_from_file_radio.isChecked())
        self.btn_train_ml.setEnabled(not busy)
        self.run_btn.setEnabled(not busy)
        self.progress_bar.setVisible(busy)
        if busy:
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat(message)
        else:
            self.progress_bar.setFormat("%p%")

    def on_run_clicked(self):
        params = self.get_parameters()
        if not params['input_path'] or not os.path.exists(params['input_path']):
            QMessageBox.warning(self, "Ошибка", "Выберите входной растр")
            return
        if not params['model_path']:
            QMessageBox.warning(self, "Внимание", "Модель не выбрана!")
            return
        self.run_requested.emit()


    def _update_db_config(self):
        if not db_storage:
            return
        db_storage.DB_CFG.update({
            'host': self.db_host_edit.text().strip(),
            'port': int(self.db_port_edit.text().strip() or 5432),
            'dbname': self.db_name_edit.text().strip(),
            'user': self.db_user_edit.text().strip(),
            'password': self.db_pass_edit.text(),
        })

    def test_db_connection(self):
        if not db_storage:
            QMessageBox.warning(self, 'Ошибка', 'db_storage не загружен')
            return
        self._update_db_config()
        try:
            conn = db_storage._conn()
            conn.close()
            self.db_status_label.setText('✓ Подключено')
            self.db_status_label.setStyleSheet('color: #2C5282; font-style: italic;')
            self.refresh_models_list()
        except Exception as e:
            self.db_status_label.setText(f'✗ {e}')
            self.db_status_label.setStyleSheet('color: #c62828; font-style: italic;')

    def refresh_models_list(self):
        if not db_storage:
            return
        self.models_list_widget.clear()
        self._update_db_config()
        try:
            for row in db_storage.list_models():
                mid, name, desc, created, training_area_wkt = row
                has_area = "📍" if training_area_wkt else ""
                text = f"[{mid}] {name} {has_area}| {created}"
                item = QListWidgetItem(text)
                item.setData(Qt.UserRole, mid)
                item.setToolTip(desc or 'Без описания')
                self.models_list_widget.addItem(item)
            self.db_status_label.setText(f'✓ Моделей: {self.models_list_widget.count()}')
        except Exception as e:
            self.db_status_label.setText(f'✗ {e}')
            self.db_status_label.setStyleSheet('color: #c62828; font-style: italic;')

    def on_btn_db_load_clicked(self):
        self.load_model_from_db(None)

    def load_model_from_db(self, item=None):
        """Загрузка модели из БД (защита от повторных вызовов)."""
        if not db_storage:
            QMessageBox.warning(self, 'Ошибка', 'Модуль db_storage не загружен')
            return
    
        # Защита от повторных вызовов
        if self._is_loading_model:
            return
    
        if isinstance(item, bool):
            item = None
        if item is None:
            item = self.models_list_widget.currentItem()
        if item is None:
            QMessageBox.warning(self, 'Внимание', 'Выберите модель в списке.')
            self._is_loading_model = True  # ← БЛОКИРОВКА
            return
        if not isinstance(item, QListWidgetItem):
            return
    
        mid = item.data(Qt.UserRole)
        if mid is None:
            return
    
        
        self._update_db_config()
        out_path = os.path.join(tempfile.gettempdir(), f'ml_model_{mid}.joblib')
    
        try:
            db_storage.load_model(mid, out_path)
            self.current_ml_model_path = out_path
            self.pretrained_path_edit.setText(out_path)
            self.load_pretrained_radio.setChecked(True)
            self.ml_status_label.setText(f'✓ Загружена из БД: {item.text()}')
            self.ml_status_label.setStyleSheet('color: #2C5282; font-style: italic;')
            self.db_status_label.setText(f'✓ Загружена модель id={mid}')
            self.db_status_label.setStyleSheet('color: #2C5282; font-style: italic;')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка загрузки', str(e))
            self.db_status_label.setText(f'✗ Ошибка: {e}')
            self.db_status_label.setStyleSheet('color: #c62828; font-style: italic;')
        finally:
            self._is_loading_model = False  # ← СНЯТИЕ БЛОКИРОВКИ

    def save_current_model_to_db(self):
        if not db_storage:
            QMessageBox.warning(self, 'Ошибка', 'db_storage не загружен')
            return
        if not self.current_ml_model_path or not os.path.exists(self.current_ml_model_path):
            QMessageBox.warning(self, 'Внимание', 'Нет модели')
            return

        name, ok = QInputDialog.getText(self, 'Сохранение в БД', 'Имя модели:')
        if not ok or not name.strip():
            return
        desc, ok2 = QInputDialog.getMultiLineText(self, 'Описание', 'Описание:')
        if not ok2:
            return

        self._update_db_config()

        params = self.get_parameters()
        training_area_wkt = self._get_raster_extent_wkt_4326()

        try:
            mid = db_storage.save_model(
                model_path=self.current_ml_model_path,
                model_name=name.strip(),
                description=desc,
                params=params['ml_params'],
                training_area_wkt=training_area_wkt,
            )
            self.db_status_label.setText(f'✓ Сохранено, id={mid}')
            self.refresh_models_list()
            QMessageBox.information(self, 'Готово', f'Модель сохранена (id={mid})')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', str(e))

    def save_model_to_file(self):
        if not self.current_ml_model_path or not os.path.exists(self.current_ml_model_path):
            QMessageBox.warning(self, 'Внимание', 'Нет модели')
            return

        default_name = os.path.basename(self.current_ml_model_path)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Сохранить модель', default_name, 'Joblib (*.joblib)'
        )
        if not path:
            return
        try:
            shutil.copy2(self.current_ml_model_path, path)
            QMessageBox.information(self, 'Готово', f'Сохранено:\n{path}')
        except Exception as e:
            QMessageBox.critical(self, 'Ошибка', str(e))

    def set_report_text(self, text):
        self.last_report_text = text
        self.report_text_edit.setPlainText(text)