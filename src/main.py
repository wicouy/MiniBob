# Clase: main.py
import os
from BertTraining import BertTraining  

retraining = True  # Cambiar a False para no reentrenar
script_directory = os.path.dirname(os.path.abspath(__file__))
training_instance = BertTraining(script_directory, retraining)
training_instance.maybe_train_or_load_model()

while True:
    input_question = input("Por favor, introduce tu pregunta (o escribe 'salir' para terminar): ")
    if input_question.lower() == 'salir':
        print("Saliendo del programa.")
        break

    input_context = input("Introduce el contexto para la pregunta: ")

    # Comprobar si la longitud de los tokens es aceptable para el modelo BERT
    tokens_question = training_instance.tokenizer.tokenize(input_question)
    tokens_context = training_instance.tokenizer.tokenize(input_context)
    
    # Reservamos 3 para los tokens especiales [CLS], [SEP] y [SEP]
    if len(tokens_question) + len(tokens_context) > 509:  
        print("La combinación de pregunta y contexto es demasiado larga para el modelo. Por favor, hazla más corta.")
        continue
    
    result = training_instance.predict(input_question, input_context)
    print("Resultado de la predicción:", result)
