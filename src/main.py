import os
import json

# Variable para controlar el reentrenamiento
retraining = True  # Cambia a False si no quieres reentrenar

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        if file_path.endswith('.jsonl'):
            return [json.loads(line) for line in file]
        else:
            return json.load(file)

script_directory = os.path.dirname(os.path.abspath(__file__))

# Leer archivos JSON y metadatos
corrections_path = os.path.join(script_directory, "data", "docs", "corrections.jsonl")
tags_path = os.path.join(script_directory, "data", "docs", "tags.json")
metadata_path = os.path.join(script_directory, "data", "docs", "data.txt")

print("Cargando datos...")
corrections_data = read_json_file(corrections_path)
tags_data = read_json_file(tags_path)
metadata_text = read_text_file(metadata_path)
print("Datos cargados.")

print(corrections_data)
