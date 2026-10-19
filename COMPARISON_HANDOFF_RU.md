# Сообщение команде и контекст для Codex

Моя часть — Elena Tuturina, роль 2 (Comparison Experiment) — выполнена.

Репозиторий: https://github.com/1r444444/CV-P22-building-segmentation

Ветка моей части: `experiment/bce-dice-comparison`. После merge берите изменения
из актуального `main`. Основная инструкция — `COMPARISON_RUN.md`.

Реализован и обучен второй вариант: тот же U-Net + ResNet18 с ImageNet encoder,
но вместо BCE используется **BCEWithLogitsLoss + soft Dice loss** с коэффициентами
1 и 1, smoothing 1. Dice считается отдельно по каждому патчу и усредняется по batch,
включая пустые маски. В loss используются вероятности без порога.

Данные, split-файлы, preprocessing, архитектура и основные настройки baseline
сохранены: Massachusetts Buildings, 4,932 / 144 / 360 патчей по 256 × 256,
batch size 8, Adam, learning rate 0.001, 10 эпох, seed 42, threshold 0.5.
Лучший checkpoint выбран на эпохе 10 по минимальному validation combined loss.
Baseline повторно не обучался. Его запуск был на T4, мой — на RTX 5070;
это ограничение сравнения указано в репорте.

| Split | Метрика | Baseline BCE | Мой BCE + Dice |
| --- | --- | --- | --- |
| Validation | IoU | 0.6338 | 0.6160 |
| Validation | Dice | 0.7759 | 0.7624 |
| Test | IoU | 0.6898 | 0.6681 |
| Test | Dice | 0.8164 | 0.8010 |

Добавление Dice в этой конфигурации не улучшило результат: test IoU снизился на
2.17 процентного пункта, Dice — на 1.54. Это результат одного запуска, а не вывод
о том, что Dice loss в целом хуже. Численные значения BCE loss и BCE + Dice loss
напрямую сравнивать нельзя. Обучение вместе с validation и сохранением checkpoint
заняло 312.19 секунды; test evaluation — 1.49 секунды с учётом загрузки и метрик.

Что читать Codex для продолжения:

- `COMPARISON_RUN.md` — точные настройки, зависимости, команды запуска и загрузки артефактов;
- `notebooks/05_comparison_bce_dice.ipynb` — выполненный notebook с выводами;
- `src/losses.py` — новая функция потерь; `src/comparison_support.py` — проверки и сведения о запуске;
- `results/metrics/comparison_metrics.csv` и `comparison_training_history.csv` — итоговые метрики и история;
- `results/metrics/comparison_initial_comparison.csv` — краткое сравнение с baseline;
- `results/metrics/comparison_run.json` и `comparison_environment.txt` — параметры, версии, SHA-256 и время;
- `results/metrics/comparison_selected_predictions.csv` — точные имена восьми примеров;
- `results/predictions/comparison_prediction_00.png` … `_07.png` и `figures/comparison_examples.png`;
- `report/SecondApproachSubmission_P22.pdf` и `.tex` — двухстраничный milestone-репорт по второй модели.

Этот PDF — отчёт по моей части, а не общий финальный technical summary команды.
Существующий baseline, его результаты и чужие notebook не изменялись.

Checkpoint и полные вероятности доступны отдельно:
https://github.com/1r444444/CV-P22-building-segmentation/releases/tag/comparison-bce-dice-2026-10-19

В Release находятся `comparison_unet_resnet18.pt`, `val.npz`, `test.npz`,
`comparison_artifact_checksums.txt` и PDF. Большие файлы не входят в обычный
`git clone`; команды скачивания и восстановления есть в `COMPARISON_RUN.md`.
Датасет также не хранится в Git: его подготовка описана там же.

В `.npz`: `probabilities` формы N × 1 × 256 × 256 (float32), `image_paths`,
`mask_paths`, `source_images`, `patch_indices`, строго в порядке соответствующего
CSV из `data/splits/`. Ground truth нужно брать из `mask_paths`, а не из восьми
сохранённых примеров. Вероятности уже прошли sigmoid. По этим файлам можно считать
метрики и анализировать пороги без GPU и повторного обучения. Для текущего результата
не нужно запускать весь notebook 05: новый запуск перезапишет его артефакты.

Ренате (роль 3): моя модель готова для общей оценки, threshold sensitivity и минимум
трёх failure cases — тени, маленькие здания, плотная застройка. Для аналогичного
анализа baseline нужно получить у Кристины её checkpoint или полные validation/test
вероятности: в репозитории baseline checkpoint отсутствует, а восьми PNG недостаточно
для оценки всего split. Выбор модели и порога делать по validation, test использовать
для финальной оценки; test не использовать для подбора настроек.

Ирине (роль 4): можно интегрировать результаты в общий README, environment,
финальный двухстраничный technical summary и demo. В общих README, data/README.md
и reproducibility checklist ещё есть исходные TBD/незаполненные пункты — их нужно
согласовать с фактическим состоянием обеих моделей. Threshold sensitivity и полный
failure analysis в моей части не выполнялись: они остаются задачами роли 3.

Проверки выполнены: все 5,436 изображений и масок патчей совпали с ожидаемыми
фрагментами raw-данных, исходные split сохранены без пересечения исходных снимков;
загрузка checkpoint заново точно воспроизвела все 504 карты вероятностей, loss,
IoU и Dice; проверены SHA-256, выбранные маски, все ячейки notebook, формула loss,
её градиенты и четыре теста. PDF собран и проверен, в нём 2 страницы.

Перед тем как отправлять ссылку на GitHub преподавателям или передавать репозиторий
другому Codex, просмотрите состав файлов и удалите явно лишние временные файлы:
черновые копии, логи сборки, editor backup, cache и временные notebook. Проверьте
`git status` и список отслеживаемых файлов: `.gitignore` не убирает уже отслеживаемые
файлы. Сохраните нужные исходники, split CSV, выполненные notebook, метрики,
предсказания, финальный PDF и его `.tex`; большие веса и вероятности берите из Release.
