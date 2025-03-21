// Código Scilab - Red Neuronal para predicción de enfermedad cardíaca
clear;
clc;
stacksize('max');

// ---------- Simulación de Datos (100 muestras) ----------
n_samples = 100;

// Variables numéricas
Edad = grand(n_samples, 1, 'uin', 29, 77);
Presion = grand(n_samples, 1, 'uin', 90, 200);
Colesterol = grand(n_samples, 1, 'uin', 150, 300);
FreqMax = grand(n_samples, 1, 'uin', 70, 200);
OldPeak = grand(n_samples, 1, 'def', 0, 6); // continuo
Vasos = grand(n_samples, 1, 'uin', 0, 3);

// Variables simbólicas (codificadas como enteros)
Sexo = grand(n_samples, 1, 'uin', 0, 1);         // 0: mujer, 1: hombre
TipoDolor = grand(n_samples, 1, 'uin', 1, 4);    // 1-4
Azucar = grand(n_samples, 1, 'uin', 0, 1);       // 0 o 1
ECG = grand(n_samples, 1, 'uin', 1, 3);          // 1-3
Angina = grand(n_samples, 1, 'uin', 0, 1);       // 0 o 1
Pendiente = grand(n_samples, 1, 'uin', 1, 3);    // 1-3
Tal = grand(n_samples, 1, 'uin', 1, 3);          // 1-3

// Clase (0: saludable, 1-3: enfermedad)
Clase = grand(n_samples, 1, 'uin', 0, 3);

// ---------- Preprocesamiento: Normalización ----------
normalize = @(x) (x - min(x)) ./ (max(x) - min(x));

NormEdad = normalize(Edad);
NormPresion = normalize(Presion);
NormColesterol = normalize(Colesterol);
NormFreqMax = normalize(FreqMax);
NormOldPeak = normalize(OldPeak);
NormVasos = normalize(Vasos);

// ---------- Construcción del conjunto de entradas ----------
X = [NormEdad NormPresion NormColesterol NormFreqMax NormOldPeak NormVasos ...
     Sexo TipoDolor Azucar ECG Angina Pendiente Tal]; // (100 x 13)

// ---------- Construcción de la salida ----------
Y = double(Clase >= 1); // Binaria: 0 = saludable, 1 = enfermedad

// Transponer matrices para la red
X = X'; // (13 x 100)
Y = Y'; // (1 x 100)

// ---------- Creación y entrenamiento de la Red Neuronal ----------
[inputs, samples] = size(X); // inputs = 13, samples = 100

// Estructura: 13 entradas -> 10 ocultas -> 1 salida
net = ann_FF_init([inputs, 10, 1], 'tanh', 'tanh');

// Entrenamiento
[net, err] = ann_FF_train(net, X, Y, 5000, 0.01);

// ---------- Pruebas y resultados ----------
Y_pred = ann_FF_run(net, X);
Y_pred_bin = (Y_pred > 0.5);

// Calcular exactitud
accuracy = sum(Y_pred_bin == Y) / samples * 100;
disp("Precisión de la red: " + string(accuracy) + "%");
