import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
import io
import os

class RadiografiaClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Clasificador de Radiografías")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Configurar estilo
        self.setup_styles()
        
        # Cargar el modelo pre-entrenado
        self.model = self.load_model()
        if not self.model:
            return
        
        # Configuración de la interfaz
        self.setup_ui()
        
    def setup_styles(self):
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 12), padding=6)
        style.configure('TLabel', font=('Helvetica', 11))
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Result.TLabel', font=('Helvetica', 14))
        
    def load_model(self):
        model_path = 'radiografia_classifier.h5'
        if not os.path.exists(model_path):
            messagebox.showerror("Error", 
                f"No se encontró el modelo '{model_path}'. Asegúrate de que esté en el mismo directorio.")
            self.root.destroy()
            return None
        
        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:
            messagebox.showerror("Error", 
                f"No se pudo cargar el modelo:\n{str(e)}")
            self.root.destroy()
            return None
    
    def setup_ui(self):
        # Frame principal con scrollbar
        main_frame = ttk.Frame(self.root)
        main_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        # Canvas y scrollbar
        self.canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Contenido de la interfaz
        self.create_widgets()
        
    def create_widgets(self):
        # Título
        title_label = ttk.Label(
            self.scrollable_frame, 
            text="Clasificador de Radiografías", 
            style='Title.TLabel'
        )
        title_label.pack(pady=(0, 20))
        
        # Botón para cargar imagen
        load_btn = ttk.Button(
            self.scrollable_frame,
            text="Cargar Imagen",
            command=self.load_image
        )
        load_btn.pack(pady=10)
        
        # Área para mostrar la imagen con tamaño fijo
        image_frame = ttk.Frame(self.scrollable_frame, relief='sunken', borderwidth=2)
        image_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        self.image_label = ttk.Label(image_frame)
        self.image_label.pack(padx=10, pady=10)
        
        # Frame para resultados
        result_frame = ttk.Frame(self.scrollable_frame)
        result_frame.pack(fill=tk.X, pady=(20, 10))
        
        ttk.Label(
            result_frame, 
            text="Resultado:", 
            style='Result.TLabel'
        ).pack(side=tk.LEFT)
        
        self.result_label = ttk.Label(
            result_frame,
            text="",
            style='Result.TLabel',
            foreground='#333333'
        )
        self.result_label.pack(side=tk.LEFT, padx=10)
        
        # Información adicional
        info_label = ttk.Label(
            self.scrollable_frame,
            text="Carga una imagen para determinar si es una radiografía o no.\nFormatos soportados: JPG, JPEG, PNG",
            style='TLabel'
        )
        info_label.pack(pady=(20, 10))
        
        # Barra de estado
        self.status_var = tk.StringVar()
        self.status_var.set("Listo")
        
        status_bar = ttk.Label(
            self.scrollable_frame,
            textvariable=self.status_var,
            relief='sunken',
            anchor=tk.W
        )
        status_bar.pack(fill=tk.X, pady=(20, 0))
        
    def load_image(self):
        file_types = [
            ("Imágenes", "*.jpg *.jpeg *.png"), 
            ("Todos los archivos", "*.*")
        ]
        
        file_path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=file_types
        )
        
        if not file_path:
            return
            
        self.status_var.set("Procesando imagen...")
        self.root.update()
        
        try:
            # Mostrar la imagen optimizada para cualquier tamaño
            self.display_image(file_path)
            
            # Preprocesar y predecir
            prediction = self.predict_image(file_path)
            
            # Mostrar resultado
            self.show_prediction_result(prediction)
            
            self.status_var.set("Imagen procesada correctamente")
            
        except Exception as e:
            self.status_var.set("Error al procesar la imagen")
            messagebox.showerror("Error", f"No se pudo procesar la imagen:\n{str(e)}")
    
    def display_image(self, file_path):
        # Leer y redimensionar la imagen manteniendo aspecto
        img = Image.open(file_path)
        
        # Calcular nuevo tamaño manteniendo aspecto
        canvas_width = self.canvas.winfo_width() - 40
        canvas_height = 500  # Altura máxima para la imagen
        
        img_ratio = img.width / img.height
        new_width = min(canvas_width, int(canvas_height * img_ratio))
        new_height = min(canvas_height, int(canvas_width / img_ratio))
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convertir para mostrar en Tkinter
        img_tk = ImageTk.PhotoImage(img)
        
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk  # Mantener referencia
        
        # Ajustar el canvas después de mostrar la imagen
        self.canvas.yview_moveto(0)
    
    def predict_image(self, image_path):
        # Preprocesamiento de la imagen para el modelo
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((224, 224), Image.LANCZOS)  # Tamaño esperado por el modelo
        
        # Convertir a array numpy y normalizar
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
        
        # Realizar predicción
        return float(self.model.predict(img_array)[0][0])
    
    def show_prediction_result(self, prediction):
        confidence = prediction if prediction >= 0.5 else (1 - prediction)
        confidence_percent = confidence * 100
        
        if prediction >= 0.5:
            result_text = f"Es una RADIOGRAFÍA (Confianza: {confidence_percent:.1f}%)"
            color = "#2196F3"  # Azul
        else:
            result_text = f"NO es una radiografía (Confianza: {confidence_percent:.1f}%)"
            color = "#FF5722"  # Naranja
        
        self.result_label.config(text=result_text, foreground=color)

if __name__ == "__main__":
    root = tk.Tk()
    
    # Configuración para escalado en Windows
    if root.tk.call('tk', 'windowingsystem') == 'win32':
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    
    app = RadiografiaClassifierApp(root)
    root.mainloop()