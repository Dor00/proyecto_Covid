import telegram
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, CallbackContext
from telegram.ext import filters
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import cv2
import io
import os
from preprocesamiento_mejorado import preprocesar_imagen
import tempfile
from telegram import Update

# Configuración
TOKEN = '8082635233:AAGytbAsxhw7B4nPg6LGHMA5T0bKsMUgfsU'
COVID_MODEL_PATH = 'modelo_covid.h5'
RADIOGRAFIA_MODEL_PATH = 'radiografia_classifier.h5'

# Cargar modelos
try:
    covid_model = load_model(COVID_MODEL_PATH)
    radiografia_model = load_model(RADIOGRAFIA_MODEL_PATH)
    print("Modelos cargados exitosamente")
except Exception as e:
    print(f"Error al cargar los modelos: {e}")
    covid_model = None
    radiografia_model = None

async def start(update: telegram.Update, context: CallbackContext):
    await update.message.reply_text(
        '¡Hola! Envíame una radiografía de tórax para:\n'
        '1. Validar si es una radiografía válida\n'
        '2. Analizar signos posibles de COVID'
    )

async def analyze_image(update: telegram.Update, context: CallbackContext):
    if not update.message or not update.message.photo:
        await update.message.reply_text('Por favor, envíame una imagen de una radiografía.')
        return

    if covid_model is None or radiografia_model is None:
        await update.message.reply_text('Lo siento, los modelos no se han cargado correctamente. Inténtalo más tarde.')
        return

    # Descargar la imagen
    photo_file = await update.message.photo[-1].get_file()
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        await photo_file.download_to_drive(tmp_file.name)
        temp_file_path = tmp_file.name

    try:
        # Paso 1: Validar si es radiografía
        is_radiografia = await validate_radiografia(temp_file_path)
        
        if not is_radiografia:
            await update.message.reply_text(
                '⚠️ La imagen no parece ser una radiografía válida. '
                'Por favor envía una imagen clara de una radiografía de tórax.'
            )
            os.unlink(temp_file_path)
            return

        # Paso 2: Analizar COVID si es radiografía
        await analyze_covid(update, temp_file_path)

    except Exception as e:
        await update.message.reply_text(f'Error al procesar la imagen: {str(e)}')
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

async def validate_radiografia(image_path):
    """Valida si la imagen es una radiografía"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = radiografia_model.predict(img_array)
        return float(prediction[0][0]) >= 0.5
    except Exception as e:
        print(f"Error validando radiografía: {e}")
        return False

async def analyze_covid(update, image_path):
    """Analiza la imagen para detectar COVID-19"""
    try:
        # Preprocesar imagen para modelo COVID
        imagen_preprocesada = preprocesar_imagen(image_path)
        
        if imagen_preprocesada is None or imagen_preprocesada.shape != (224, 224, 1):
            await update.message.reply_text('La imagen no tiene las dimensiones esperadas para el análisis.')
            return

        imagen_preprocesada = np.expand_dims(imagen_preprocesada, axis=0)
        
        # Predecir COVID
        pred = covid_model.predict(imagen_preprocesada)
        prob_covid = pred[0][1] * 100

        # Generar respuesta
        if prob_covid > 70:
            respuesta = (
                "⚠️⚠️ ALTA PROBABILIDAD DE COVID-19 ⚠️⚠️\n"
                f"Probabilidad: {prob_covid:.2f}%\n\n"
                "Recomendación: Consulta inmediatamente con un especialista."
            )
        elif prob_covid > 50:
            respuesta = (
                "⚠️ Posible COVID-19 detectado\n"
                f"Probabilidad: {prob_covid:.2f}%\n\n"
                "Recomendación: Consulta con un médico para evaluación adicional."
            )
        else:
            respuesta = (
                "✅ No se detectaron signos claros de COVID-19\n"
                f"Probabilidad: {prob_covid:.2f}%\n\n"
                "Nota: Este resultado no descarta completamente la posibilidad de infección. "
                "Consulta a un médico si tienes síntomas."
            )

        await update.message.reply_text(respuesta)

    except Exception as e:
        await update.message.reply_text(f'Error en el análisis de COVID: {str(e)}')
        raise

def main():
    application = ApplicationBuilder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, analyze_image))
    
    print("Bot iniciado...")
    application.run_polling()

if __name__ == '__main__':
    main()