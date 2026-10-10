# ai_agent

Интерактивный чат-агент (`agent.py`) через OpenAI-совместимый API с TOOL
`run_shell` (полноценный shell), логами API/инструментов и сохранением истории.
Конфигурация — `../config.yaml` (пример в начале `agent.py`).
Модульные тесты — `test_agent.py` (`python -m pytest test_agent.py -p no:libtmux`).

## TODO — выбор модели во время сессии

Планируемое расширение (подробные маркеры `TODO(...)` — в docstring модуля
и функции `main()` в `agent.py`):

* `TODO(model-switch)` — команда `/model <имя>`: смена текущей модели без
  перезапуска (переменная `model` читается динамически в каждом запросе).
* `TODO(model-list)` — команда `/models`: список доступных моделей через
  `client.models.list()` (GET /models) с обработкой неподдерживаемого эндпоинта.
* `TODO(model-validate)` — проверка выбранной модели по списку сервера
  (предупреждение, но не блокировка, если /models недоступен).
* `TODO(model-history)` — служебная запись смены модели в `chat_history.jsonl`.
* `TODO(model-log)` — запись смены модели в `api.log` (обрезка до 20 символов).
* `TODO(model-config-fallback)` — опциональный `allowed_models` в config.yaml
  для ограничения выбора.
* `TODO(tests)` — модульные тесты нового поведения (парсинг `/model`,
  мок `models.list()`, записи в историю/лог).
