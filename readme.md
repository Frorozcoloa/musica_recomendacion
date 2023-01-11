# Clasificación y recomendación de canciones

Empresas como Soptify y Youtube, su principal negoció es recomendar canciones, dependiendo. También es bueno clasificar las canciones para mejorar la experiencia de usuario.

Para solucionar este problema, se puede tener un enfoque de deep learning o un enfoque machine learning. La idea es hacer la extracción de características.

En este proyectos, se va hacer por medio de machine learning, extrayendo las caracteristicas del sonido.

## Estructura.

```
Data --> Carpeta donde está el proyecto
Notebooks --> Donde está el notebook del proyecto
src --> Aquí es donde se pondrá el modelo en producción
```

## Producción

La producción del modelo se va hacer en  ```streamilit```.

El modelo en producción va a tener tres partes.

    1.  Pre-procesamiento, donde se hace la extracción de caraceristicas. En el notebook 1-EDA.ipynb se explica las características que se extrayeron
    2.  Pos-procesamiento, donde se normalaizan los datos, antes de ingrear al modelo