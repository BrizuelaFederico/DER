# Escanear imagen Diagrama Entidad-Relación (DER)

Este proyecto consta en cargar una imagen que contenga un DER, y poder obtener de la misma las entidades y relaciones encontradas, como también el nombre y los atributos de dichas entidades.

Para conseguir este análisis, se utilizaron diferentes técnicas de inteligencia artificial, teniendo que construir el modelo desde cero, empleando herramientas como Tensorflow, Keras y Google Colab.  
El reconocimiento y detección del texto, se usaron modelos ya creados, en base a la documentación de [OpenCV](https://docs.opencv.org/4.x/d4/d43/tutorial_dnn_text_spotting.html).

## Formato de las imágenes que se pueden utilizar 

Debe ser de extensión JPG, y un diseño parecido al siguiente (ya que para el entrenamiento de las IAs, se usaron con similar apariencia):

![Ejemplo tipo de Imagen!](/images/example_images/example_1.jpg "Ejemplo tipo de imagen")

## Ejemplo del resultado que muestra la aplicación

![Entidades y relaciones encontradas!](/images/app_images/entities_and_relationships_found.png "Entidades y relaciones encontradas")

![Nombre y atributos!](/images/app_images/name_and_attributes.png "Nombre y atributos")

![Relaciones entre entidades!](/images/app_images/relationships_between_entities.png "Relaciones entre entidades")

## Ejecutar la aplicación

Es necesario tener instalado Python (la versión utilizada es la 3.12), y haber descargado este repositorio.
Luego, dentro de la carpeta principal del proyecto (donde se encuentra el archivo requirementes.txt), ejecutar los siguientes comandos:


Instalar las dependencias: (recordar que es conveniente haber creado y activado el entorno virtual)
```
pip install -r requirements.txt
```

Levantar servidor: (hay que estar dentro de la carpeta app)
```
uvicorn src.main:app --port 8000
```

Una vez levantado, abrir un navegador, e ir a http://127.0.0.1:8000

## Ejecutar la aplicación con Docker

Estando en el directorio donde se encuentra el archivo Dockerfile

Crear imagen:
```
docker build -t der .
```

Correr el contenedor a partir de la imagen creada:
```
docker run -it -p 8000:8000 --rm der
```

Una vez levantado, solo queda por ir a http://127.0.0.1:8000









