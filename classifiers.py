# -*- coding: utf-8 -*-
"""
ML-классификатор (Random Forest, SVM) и утилиты для стилей/отчётов.
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from qgis.core import (
    QgsRasterLayer, QgsRasterFileWriter, QgsRectangle,
    Qgis, QgsMessageLog
)


# ==================== УТИЛИТЫ: СТИЛИ И ОТЧЁТЫ ====================
class ClassificationUtils:
    """Общие утилиты: стили, отчёты, коды классов."""

    CLASS_CODES = {
        'background': 0, 'oil': 1, 'vegetation': 2,
        'roads': 3, 'infrastructure': 4,
    }
    CLASS_NAMES = {
        0: 'Фон', 1: 'Нефть/Нефтепродукты', 2: 'Растительность',
        3: 'Дороги', 4: 'Инфраструктура',
    }
    CLASS_COLORS = {
        0: (0, 0, 0, 255),
        1: (30, 30, 30, 255),
        2: (34, 139, 34, 255),
        3: (139, 119, 101, 255),
        4: (178, 34, 34, 255),
    }

    @staticmethod
    def apply_style(layer):
        """Применяет палитру к слою классификации."""
        from qgis.PyQt.QtGui import QColor
        from qgis.core import QgsPalettedRasterRenderer

        provider = layer.dataProvider()
        class_data = []
        for code, (r, g, b, a) in ClassificationUtils.CLASS_COLORS.items():
            if code == 0:
                continue
            class_data.append(
                QgsPalettedRasterRenderer.Class(
                    code, QColor(r, g, b, a), ClassificationUtils.CLASS_NAMES[code]
                )
            )
        layer.setRenderer(QgsPalettedRasterRenderer(provider, 1, class_data))
        layer.triggerRepaint()

    @staticmethod
    def generate_report_text(classification_result_path: str,
                             source_input_path: str = "") -> str:
        """Генерация текста отчёта (возвращает строку, не сохраняет в файл)."""
        if not classification_result_path or not os.path.exists(classification_result_path):
            return ""

        result_layer = QgsRasterLayer(classification_result_path, "temp_result")
        if not result_layer.isValid():
            return ""

        provider = result_layer.dataProvider()
        width, height = result_layer.width(), result_layer.height()
        extent = result_layer.extent()
        total_pixels = width * height

        counts = {code: 0 for code in ClassificationUtils.CLASS_CODES.values() if code != 0}
        block_size = 512

        for r in range(0, height, block_size):
            h = min(block_size, height - r)
            for c in range(0, width, block_size):
                w = min(block_size, width - c)
                rect = QgsRectangle(
                    extent.xMinimum() + c * extent.width() / width,
                    extent.yMaximum() - (r + h) * extent.height() / height,
                    extent.xMinimum() + (c + w) * extent.width() / width,
                    extent.yMaximum() - r * extent.height() / height
                )
                block = provider.block(1, rect, w, h)
                data = np.frombuffer(block.data(), dtype=np.uint8).reshape((h, w))
                for code in counts.keys():
                    counts[code] += int(np.sum(data == code))

        lines = [
            "=" * 50,
            "ОТЧЁТ О ML-КЛАССИФИКАЦИИ",
            "=" * 50,
            f"\nФайл результата: {classification_result_path}",
            f"Исходный растр:   {source_input_path}",
            f"Размер:           {width} x {height} пикселей",
            f"Всего пикселей:   {total_pixels:,}",
            "\n" + "-" * 50,
            "СТАТИСТИКА ПО КЛАССАМ:",
            "-" * 50,
        ]

        for code in sorted(counts.keys()):
            count = counts[code]
            pct = (count / total_pixels) * 100 if total_pixels > 0 else 0
            lines.append(
                f"{ClassificationUtils.CLASS_NAMES.get(code, f'Класс {code}'):<25} "
                f"{count:>10,} пикс.  ({pct:6.2f}%)"
            )

        lines += ["\n" + "=" * 50]
        return "\n".join(lines)


# ==================== ML КЛАССИФИКАТОР ====================
class SklearnClassifier:
    """Обучение и предсказание ML-моделей (RF или SVM). Поблочная обработка."""

    MAX_TRAIN_PIXELS = 200000

    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)

    def train(self, raster_path: str, labels_path: str,
              model_type: str = 'rf',
              n_estimators: int = 100, max_depth=None,
              svm_kernel: str = 'rbf', svm_C: float = 1.0,
              block_size: int = 1024, progress_callback=None) -> str:
        """Поблочное обучение со случайной выборкой пикселей."""
        img = QgsRasterLayer(raster_path, "tmp_train_img")
        lbl = QgsRasterLayer(labels_path, "tmp_train_lbl")

        if not img.isValid() or not lbl.isValid():
            raise RuntimeError("Не удалось открыть растр или метки")

        p_img = img.dataProvider()
        p_lbl = lbl.dataProvider()
        bands = p_img.bandCount()
        W, H = img.width(), img.height()
        ext = img.extent()

        X_parts, y_parts = [], []
        total_blocks = ((H + block_size - 1) // block_size) * ((W + block_size - 1) // block_size)
        processed = 0

        for r in range(0, H, block_size):
            h = min(block_size, H - r)
            for c in range(0, W, block_size):
                w = min(block_size, W - c)
                rect = QgsRectangle(
                    ext.xMinimum() + c * ext.width() / W,
                    ext.yMaximum() - (r + h) * ext.height() / H,
                    ext.xMinimum() + (c + w) * ext.width() / W,
                    ext.yMaximum() - r * ext.height() / H,
                )

                stack = np.empty((bands, h, w), dtype=np.float32)
                for b in range(1, bands + 1):
                    blk = p_img.block(b, rect, w, h)
                    src_type = p_img.dataType(b)
                    dtype = np.uint16 if src_type in (Qgis.Byte, Qgis.UInt16, Qgis.Int16) else np.float32
                    stack[b - 1] = np.frombuffer(blk.data(), dtype=dtype).reshape(h, w).astype(np.float32)

                lbl_blk = p_lbl.block(1, rect, w, h)
                labels = np.frombuffer(lbl_blk.data(), dtype=np.uint8).reshape(h, w)
                mask = (labels > 0) & (labels != 255)

                if mask.any():
                    X_parts.append(stack[:, mask].T)
                    y_parts.append(labels[mask])

                processed += 1
                if progress_callback and processed % 5 == 0:
                    pct = int((processed / total_blocks) * 50)
                    progress_callback(pct, f"Сбор пикселей: {processed}/{total_blocks}")

        if not X_parts:
            raise ValueError("Нет валидных пикселей для обучения!")

        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        total = len(y)
        if total > self.MAX_TRAIN_PIXELS:
            rng = np.random.default_rng(42)
            idx = rng.choice(total, self.MAX_TRAIN_PIXELS, replace=False)
            X, y = X[idx], y[idx]
            QgsMessageLog.logMessage(
                f"Выборка: {self.MAX_TRAIN_PIXELS} из {total}", 'MultispectralClassifier'
            )

        if progress_callback:
            progress_callback(60, f"Обучение {model_type.upper()}: {len(y)} пикселей")

        # Выбор модели
        if model_type == 'svm':
            clf = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel=svm_kernel,
                    C=svm_C,
                    class_weight='balanced',
                    random_state=42,
                    probability=True
                ))
            ])
        else:
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )

        clf.fit(X, y)
        self.model = clf

        model_filename = os.path.splitext(raster_path)[0] + f"_{model_type}_model.joblib"
        joblib.dump(clf, model_filename)
        self.model_path = model_filename

        classes_ = clf.classes_ if hasattr(clf, 'classes_') else clf.named_steps['svm'].classes_
        QgsMessageLog.logMessage(
            f"Модель сохранена: {model_filename}. Классов: {len(classes_)}",
            'MultispectralClassifier'
        )

        return model_filename

    def predict(self, raster_path: str, output_path: str,
                block_size: int = 1024, progress_callback=None) -> str:
        """Поблочное предсказание."""
        if self.model is None:
            raise RuntimeError("Модель не загружена")

        lyr = QgsRasterLayer(raster_path, "tmp_pred")
        if not lyr.isValid():
            raise RuntimeError("Не удалось открыть растр")

        p = lyr.dataProvider()
        bands = p.bandCount()
        W, H = lyr.width(), lyr.height()
        ext = lyr.extent()

        writer = QgsRasterFileWriter(output_path)
        writer.setOutputProviderKey('gdal')
        writer.setCreateOptions(['COMPRESS=LZW'])
        out = writer.createOneBandRaster(Qgis.DataType.Byte, W, H, ext, lyr.crs())

        if out is None:
            raise RuntimeError(f"Не удалось создать файл: {output_path}")

        out.setEditable(True)
        out.setNoDataValue(1, 0)

        total_blocks = ((H + block_size - 1) // block_size) * ((W + block_size - 1) // block_size)
        processed = 0

        for r in range(0, H, block_size):
            h = min(block_size, H - r)
            for c in range(0, W, block_size):
                w = min(block_size, W - c)
                rect = QgsRectangle(
                    ext.xMinimum() + c * ext.width() / W,
                    ext.yMaximum() - (r + h) * ext.height() / H,
                    ext.xMinimum() + (c + w) * ext.width() / W,
                    ext.yMaximum() - r * ext.height() / H,
                )

                stack = np.empty((bands, h, w), dtype=np.float32)
                for b in range(1, bands + 1):
                    blk = p.block(b, rect, w, h)
                    src_type = p.dataType(b)
                    dtype = np.uint16 if src_type in (Qgis.Byte, Qgis.UInt16, Qgis.Int16) else np.float32
                    stack[b - 1] = np.frombuffer(blk.data(), dtype=dtype).reshape(h, w).astype(np.float32)

                Xb = stack.reshape(bands, -1).T
                preds = self.model.predict(Xb).reshape(h, w).astype(np.uint8)

                wb = out.block(1, rect, w, h)
                wb.setData(preds.tobytes())
                out.writeBlock(wb, 1, c, r)

                processed += 1
                if progress_callback and processed % 5 == 0:
                    pct = 10 + int((processed / total_blocks) * 80)
                    progress_callback(pct, f"Предсказание: {processed}/{total_blocks}")

        out.setEditable(False)
        return output_path