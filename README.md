# EEG-sleep-stage-scoring
Sleep stage scoring based on EEG 

[Презентация](https://docs.google.com/presentation/d/1CRf0Ed_oofX6UvPUE2znrWHistZ22UPKsPr0e2InFj8/edit?usp=sharing)

В папке `notebooks` есть ноутбук бейзлайна, CNN (`cnn_encoder.ipynb`), работающей (но упрощенно) сети с lstm (`lstm_subsequences.ipynb`) и пока не доделанной более сложной сети с lstm (`lstm_shared_state.ipynb`). Также в этой папке есть `cnn_encoder.pt` (результат обучения CNN блока на 50 эпохах), который можно передавать на вход lstm.

В папке `eegproject` содержатся модели и обработка данных.

В коллабе ноутбуки с сетями запускаются на сохраненном обработанном куске датасета (потому что обработка долгая), который хранится на гугл диске. Чтобы загружать данные заново, нужно убрать `preprocessed_path` при загрузке `train` и `test` датасетов. Также для работы ноутбука нужно загрузить в коллаб zip-архив, сделанный из папки `eegproject`. А для работы lstm-ноутбуков еще и `cnn_encoder.pt`.
