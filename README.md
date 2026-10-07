# Сравнение ANN на MNIST

Проект содержит ann_mnist_benchmark.ipynb для сравнения Exact Brute Force,
FlyHash, HNSW и IVFPQ по Recall@1/5/10, времени и дополнительному пиковому RSS.

## Запуск

Требуется Python 3.11 или 3.12. Распакуйте архив и откройте терминал в папке проекта.

Создайте виртуальную среду:

    python -m venv .venv

Активируйте её:

- Linux/macOS: source .venv/bin/activate
- Windows PowerShell: .venv\Scripts\Activate.ps1
- Windows cmd: .venv\Scripts\activate.bat

Установите зависимости:

    python -m pip install -r requirements.txt

Для hnswlib может потребоваться C++11-совместимый компилятор.
Если Python настроен на отсутствующий Clang, в Linux можно выбрать GCC:

    CC=gcc CXX=g++ python -m pip install -r requirements.txt

Поместите исходный CSV MNIST в:

    data/mnist_784.csv

Файл должен содержать 784 числовых пиксельных признака в диапазоне [0, 255]
и один столбец меток цифр 0–9. Целевой столбец определяется с проверкой содержимого.
В эксперименте используются 10 000 train и 500 test с двумя этапами стратификации.

Запустите JupyterLab из папки проекта:

    python -m jupyterlab

Откройте ann_mnist_benchmark.ipynb и выполните все ячейки сверху вниз.
Первая техническая ячейка также умеет устанавливать недостающие зависимости через %pip.

## Данные и результаты

CSV MNIST в архив не включён: его не было в предоставленной среде.
Полное выполнение notebook пока не проверено: запуск остановился на отсутствии CSV.
Синтаксис и вспомогательные функции проверены; реальные экспериментальные результаты
не создавались. В архиве notebook сохранён без выводов и сообщения об отсутствии файла.

После успешного запуска в results/ появятся:

- exact_neighbors_k10.pkl
- flyhash_neighbors_k10.pkl
- hnsw_neighbors_k10.pkl
- ivfpq_neighbors_k10.pkl
- ann_results.csv
- recall_vs_k.png
- recall10_vs_search_time.png
- recall10_vs_memory.png
- split_indices.npz
- experiment_manifest.json
- summary.md
- hypotheses.json

Параметры эксперимента закреплены в notebook. Все методы используют одни и те же
нормализованные contiguous float32 train/test. FlyHash использует настоящую библиотеку,
упакованные бинарные коды и Hamming distance с воспроизводимым порядком равных расстояний.
RSS измеряется одинаковым sampler; графики и H1–H4 строятся только из полученных измерений.
