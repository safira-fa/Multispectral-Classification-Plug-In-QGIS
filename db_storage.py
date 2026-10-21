# -*- coding: utf-8 -*-
import hashlib, json
import psycopg2
from psycopg2.extras import Json

DB_CFG = {
    'host': '192.168.0.0',
    'port': 5432,
    'dbname': '',
    'user': '',
    'password': '',
}

def _conn():
    return psycopg2.connect(**DB_CFG)

def save_model(model_path, model_name, description, params, training_area_wkt=None):
    """Сохранение модели + (опционально) геометрии области обучения."""
    with open(model_path, 'rb') as f:
        data = f.read()
    with _conn() as conn, conn.cursor() as cur:
        if training_area_wkt:
            cur.execute("""
                INSERT INTO ml_models (model_name, description, params_json, model_bytes, training_area)
                VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 4326))
                RETURNING id
            """, (model_name, description, Json(params), data, training_area_wkt))
        else:
            cur.execute("""
                INSERT INTO ml_models (model_name, description, params_json, model_bytes)
                VALUES (%s, %s, %s, %s) RETURNING id
            """, (model_name, description, Json(params), data))
        return cur.fetchone()[0]

def list_models():
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, model_name, description, created_at,
                   CASE WHEN training_area IS NOT NULL THEN ST_AsText(training_area) ELSE NULL END
            FROM ml_models ORDER BY created_at DESC
        """)
        return cur.fetchall()

def load_model(model_id, out_path):
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT model_bytes FROM ml_models WHERE id=%s", (model_id,))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Модель не найдена")
        with open(out_path, 'wb') as f:
            f.write(row[0])
        return out_path

def find_models_in_extent(minx, miny, maxx, maxy, epsg=4326):
    """PostGIS-запрос: найти модели, обученные на территории, пересекающейся с заданным экстентом."""
    with _conn() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, model_name, description
            FROM ml_models
            WHERE training_area IS NOT NULL
              AND ST_Intersects(training_area, ST_MakeEnvelope(%s, %s, %s, %s, %s))
        """, (minx, miny, maxx, maxy, epsg))
        return cur.fetchall()