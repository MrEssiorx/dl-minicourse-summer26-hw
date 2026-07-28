# LCNN Anti-Spoofing (ASVspoof2019 LA)

Реализация и обучение системы противодействия спуфингу (Countermeasure) на
Logical Access (LA) партиции датасета ASVspoof2019. Задача  бинарная
классификация записей на `bonafide` (настоящий голос) и `spoof`
(spoof-атаки). Модель --- LightCNN (LCNN), 
вход --- STFT-спектрограмма, метрика качества --- Equal Error Rate (EER).

Код является ответвлением [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## Результат

На eval-партиции LA достигнут **EER = 7.2%**

## Ключевые гиперпараметры

| Компонент | Значение |
| --- | --- |
| Front-end | STFT, окно 20 мс, шаг 10 мс, n_fft 512 → 257 частотных бинов, FixedLengthCrop до 750 фреймов (zero-pad коротких, случайная обрезка длинных) |
| Модель | LightCNN, dropout 0.75 (3.93M параметров) |
| Оптимизатор | Adam, lr 3e-4 |
| LR-scheduler | StepLR, ×0.9 каждую эпоху |
| Batch size | 64 |
| Эпохи | 10 (epoch_len 500 шагов) |
| Функция потерь | Cross-Entropy с весами классов `[0.557, 4.919]` (spoof, bonafide) |
| Random Seed | 1 |

## Структура проекта

Написано с нуля (LCNN / ASVspoof):

- `src/model/lcnn.py`, `src/model/mfm.py` --- модель и MFM-активация;
- `src/datasets/asvspoof_dataset.py` --- датасет и построение индекса по протоколу;
- `src/loss/cross_entropy.py` --- функция потерь;
- `src/metrics/eer.py` --- метрика EER;
- `src/transforms/stft.py`, `src/transforms/fixed_length.py` --- front-end;
- `src/configs/**` --- файлы конфигурации

Адаптированные файлы шаблона: `src/datasets/base_dataset.py` (загрузка аудио +
front-end), `src/datasets/collate.py`, `src/trainer/*`. Остальное --- шаблон без
изменений.

## Воспроизведение


#### Kaggle-ноутбук

`<ссылка на Kaggle-ноутбук>`

#### На локальной машине:

0. Установить зависимости:
    ```bash
    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    ```

1. Скачать ASVspoof2019 LA ([datashare](https://datashare.ed.ac.uk/handle/10283/3336)
   или [Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)).

2. Обучение (создаёт `saved/lcnn/checkpoint-epoch10.pth`):

   ```bash
   python train.py data_dir=/путь/к/LA/LA
   ```


   Путь `data_dir` должен указывать на директорию, содержащую
   `ASVspoof2019_LA_train`, `ASVspoof2019_LA_dev`, `ASVspoof2019_LA_eval`,
   `ASVspoof2019_LA_cm_protocols`.

3. Построение прогноза на eval (создаёт `data/saved/asvspoof/eval.csv`)
   ```bash
   python inference.py data_dir=/путь/к/LA/LA
   ```

## Кредиты

- Шаблон: [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template),
  Petr Grinberg (Blinorot).
- LightCNN: Wu et al., [arXiv:1511.02683](https://arxiv.org/abs/1511.02683).
- LCNN для anti-spoofing (STC): Lavrentyeva et al., [arXiv:1904.05576](https://arxiv.org/abs/1904.05576).
- Recipe и сравнение лоссов: Wang & Yamagishi, [arXiv:2103.11326](https://arxiv.org/abs/2103.11326).

