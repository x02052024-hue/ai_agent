#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent.py — интерактивный чат-агент через OpenAI-совместимый API.

НАЗНАЧЕНИЕ
----------
Скрипт запускает интерактивный диалог с языковой моделью (LLM), доступ к
которой осуществляется по OpenAI-совместимому HTTP API (OpenAI, vLLM,
Ollama, llama.cpp server, LocalAI и т.п.). Ключ и URL сервера читаются из
файла ``../config.yaml`` (на уровень выше самого скрипта); пример конфигурации
приведён текстовым блоком в начале этого файла — см. константу EXAMPLE_CONFIG.

ВОЗМОЖНОСТИ
-----------
  * Интерактивный режим: ввод пользователя -> запрос к модели -> печатный ответ,
    цикл до команды выхода (exit / quit / выход).
  * Поддержка TOOL с полноценным shell: модели в каждом запросе передаётся
    описание функции ``run_shell`` (function_call). Если модель вызывает её,
    команда выполняется локально через bash (или sh), а stdout/stderr/код
    возврата отправляются обратно как сообщение role="tool".
  * Лог API-взаимодействий (``api.log``): каждая реплика пользователя и модели
    записывается с временной меткой; текст ограничивается 20 символами.
  * Системный промпт запрещает ответы с нечитаемыми символами при кодировке
    UTF-8 (русский язык должен отображаться корректно).
  * История чата сохраняется после каждого хода в отдельный файл
    ``chat_history.jsonl`` (формат JSON Lines, одна строка = одно сообщение).
  * Журнал использования TOOL (``tool_usage.log``): время, имя инструмента,
    аргументы, код возврата, предпросмотр результата.
  * В начале сессии (и в каждом последующем запросе) через API передаётся
    описание TOOL — схема function_call RUN_SHELL_TOOL.

ПРЕДСТАВЛЕННЫЕ ФАЙЛЫ (относительно каталога скрипта)
----------------------------------------------------
  ../config.yaml       — конфигурация (api_key, base_url, model); НЕ создаётся
                         автоматически, пользователь готовит по примеру ниже.
  api.log              — лог API-взаимодействий (дополняется).
  tool_usage.log       — журнал вызовов инструментов (дополняется).
  chat_history.jsonl   — история текущего чата (перезаписывается целиком).

БЫСТРЫЙ СТАРТ
-------------
    1. Создайте ../config.yaml по примеру EXAMPLE_CONFIG (см. ниже).
    2. pip install -r requirements.txt
    3. python agent.py
    4. Команды сессии: 'exit'/'quit'/'выход' — завершение,
       'clear' — очистка истории, всё остальное — сообщение модели.

TODO — ПЛАНИРУЕМЫЕ ВОЗМОЖНОСТИ
------------------------------
Выбор модели во время сессии (горячая замена модели без перезапуска):

    TODO(model-switch): Реализовать команду '/model <имя>' в интерактивном
        цикле main(): менять значение переменной current_model, которая
        передаётся в chat.completions.create(). Текущая модель уже вынесена
        в локальную переменную `model`, достаточно читать её динамически.

    TODO(model-list): Реализовать команду '/models': получить список доступных
        моделей сервера через client.models.list() (стандартная эндпоинта
        OpenAI-совместимых серверов GET /models) и напечатать его. Обработать
        случай, когда сервер не поддерживает /models (404/501) — вежливое
        сообщение вместо трейсбека.

    TODO(model-validate): Перед переключением проверять выбранную модель по
        списку с сервера (если он доступен); при отсутствии — предупреждать,
        но разрешить переключение (некоторые серверы не отдают /models).

    TODO(model-history): Помечать в chat_history.jsonl момент смены модели
        (например, служебной записью {"role": "system_note", "event":
        "model_switch", "from": ..., "to": ...}), чтобы историю можно было
        воспроизвести с той же последовательностью моделей.

    TODO(model-log): Писать смену модели в api.log: log_api("model_switch",
        f"{old} -> {new}") — с тем же ограничением 20 символов.

    TODO(model-config-fallback): Если в config.yaml задан список
        allowed_models, ограничить выбор командой '/model' только им;
        по умолчанию — любые имена.

    TODO(tests): Модульные тесты для нового поведения: парсинг команды
        '/model', вызов models.list() с мок-клиентом, запись model_switch
        в историю и лог, отказ на некорректное имя при включённом
        allowed_models.

TODO — ОБЩИЕ ПЛАНЫ (вне выбора модели)
--------------------------------------
    TODO(streaming): Поддержка потоковой генерации (stream=True) с выводом
        токенов по мере поступления.
    TODO(confirm-shell): Опциональное подтверждение пользователем каждой
        shell-команды перед выполнением (режим безопасности).
    TODO(rotate-logs): Ротация api.log / tool_usage.log по размеру.
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from urllib.parse import urlparse

import httpx
import yaml
from openai import APIConnectionError, APITimeoutError, OpenAI

#: Текстовый блок с описанием и примером конфигурации, который размещён
#: в начале модуля и печатается пользователю при проблемах с config.yaml.
EXAMPLE_CONFIG = """\
# Пример ../config.yaml:
api_key: "sk-my-secret-key"
base_url: "http://localhost:8000/v1"
model: "gpt-4o-mini"
"""

# ---------------------------------------------------------------------------
# Пути к файлам. Все — относительно каталога скрипта; конфиг лежит уровнем
# выше (../config.yaml), остальные артефакты — рядом со скриптом.
# Вынесены в константы: тесты подменяют их через monkeypatch.setattr(agent, ...).
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
API_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.log")
TOOL_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_usage.log")
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.jsonl")

#: Имя модели по умолчанию, если оно не задано в config.yaml.
#: TODO(model-switch): во время сессии может быть заменено командой '/model'.
DEFAULT_MODEL = "gpt-4o-mini"

#: Системный промпт. Обязательное требование (по ТЗ): запрет нечитаемых
#: символов при кодировке UTF-8 — русский текст должен быть корректным.
SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай на русском языке. "
    "Во всех ответах недопустимы нечитаемые символы при кодировке UTF-8 "
    "(русский язык должен отображаться корректно, без кракозябр и битых байтов). "
    "Для выполнения системных команд используй инструмент run_shell."
)

#: Описание TOOL в формате function_call (схема OpenAI tools API).
#: Передаётся модели в параметре ``tools`` при каждом запросе сессии —
#: фактически описание инструмента отправляется в начале сессии и
#: сопровождает каждый последующий ход диалога.
RUN_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": (
            "Выполняет переданную строку команды в полноценном shell (bash/sh) "
            "на компьютере пользователя и возвращает stdout, stderr и код возврата."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Команда shell для выполнения, например: ls -la",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Таймаут выполнения в секундах (по умолчанию 60).",
                },
            },
            "required": ["command"],
        },
    },
}

#: Таймаут выполнения shell-команды по умолчанию, секунд
#: (может быть переопределён моделью через аргумент ``timeout``).
DEFAULT_SHELL_TIMEOUT = 60


def truncate(text, limit=20):
    """Ограничить ``text`` до ``limit`` символов для записи в лог.

    Требование ТЗ: текст пользователя и модели в api.log — не более 20
    символов. Переводы строк заменяются пробелами, чтобы запись лога
    всегда занимала ровно одну строку; ``None`` трактуется как пустая
    строка.

    Args:
        text (str | None): исходный текст реплики.
        limit (int): максимальная длина результата (по умолчанию 20).

    Returns:
        str: обрезанная однострочная копия текста.
    """
    text = (text or "").replace("\n", " ")
    return text[:limit]


class ConfigError(Exception):
    """Ошибка конфигурации (отсутствует файл, битый YAML или некорректные значения)."""


def load_config(path):
    """Прочитать и проверить конфигурацию из YAML-файла ``path``.

    Порядок обработки ошибок (все — :class:`ConfigError` с человекочитаемым
    сообщением и примером конфигурации, без трейсбека):

    1. Файл отсутствует -> подсказка создать его по EXAMPLE_CONFIG.
    2. Некорректный синтаксис YAML -> подсказка исправить синтаксис.
    3. Документ не является словарью -> ошибка формата.
    4. Отсутствуют обязательные ключи ``api_key`` / ``base_url``.
    5. :func:`validate_config` — семантическая проверка base_url/model.

    Args:
        path (str): путь к config.yaml (обычно CONFIG_PATH = ../config.yaml).

    Returns:
        dict: словарь конфигурации с ключами ``api_key``, ``base_url``
        и, опционально, ``model``.

    Raises:
        ConfigError: на любую из перечисленных проблем с файлом.
    """
    if not os.path.exists(path):
        raise ConfigError(
            f"Не найден файл конфигурации: {path}\n"
            "Создайте его по примеру:\n" + EXAMPLE_CONFIG
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Файл {path} содержит некорректный YAML: {e}\n"
            "Исправьте синтаксис (частая причина — лишнее двоеточие в значении)\n"
            "или создайте файл заново по примеру:\n" + EXAMPLE_CONFIG
        ) from e
    if not isinstance(cfg, dict):
        raise ConfigError(f"Файл {path} не содержит корректных параметров.")
    for key in ("api_key", "base_url"):
        if key not in cfg:
            raise ConfigError(f"В {path} отсутствует обязательный параметр '{key}'.")
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    """Семантическая проверка конфигурации до первого обращения к API.

    Защищает от типовых ошибок пользователя:

    * перепутанные местами значения ``model`` и ``base_url``;
    * ``base_url`` без схемы http/https или без хоста;
    * URL, указанный вместо имени модели в поле ``model``.

    Args:
        cfg (dict): словарь конфигурации (после проверки обязательных ключей).

    Raises:
        ConfigError: при обнаружении любой из перечисленных проблем; сообщение
            содержит конкретные значения и пример корректного конфига.
    """
    base_url = str(cfg["base_url"]).strip()
    model = str(cfg.get("model", "")).strip()

    def starts_with_url(s):
        return s.startswith(("http://", "https://"))

    # Частая ошибка: значения base_url и model поменяли местами.
    if starts_with_url(model) and not starts_with_url(base_url):
        raise ConfigError(
            f"Похоже, в конфиге перепутаны значения 'model' и 'base_url':\n"
            f"  model: {model!r}\n"
            f"  base_url: {base_url!r}\n"
            "base_url должен начинаться с http:// или https://, а model — это имя модели.\n"
            "Исправьте файл по примеру:\n" + EXAMPLE_CONFIG
        )

    parsed = urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError(
            f"Некорректный 'base_url': {base_url!r} — должен начинаться с http:// или https://\n"
            "Пример:\n" + EXAMPLE_CONFIG
        )

    if starts_with_url(model):
        raise ConfigError(
            f"В параметре 'model' указан URL ({model!r}) вместо имени модели.\n"
            "Проверьте, не перепутаны ли значения 'model' и 'base_url'.\n"
            "Пример:\n" + EXAMPLE_CONFIG
        )


def log_api(kind, text):
    """Записать одно взаимодействие с API в ``api.log`` (требование ТЗ).

    Формат строки: ``[ISO-время] <kind>: <текст до 20 символов>``.
    ``kind`` — роль участника диалога: ``user`` (реплика пользователя),
    ``model`` (ответ модели); TODO(model-log) планирует также ``model_switch``.
    Текст ограничивается :func:`truncate` до 20 символов — это явное требование
    задания на логирование пользовательского и модельного текста.

    Args:
        kind (str): тип записи ('user' / 'model' / ...).
        text (str | None): исходный текст реплики (будет обрезан).
    """
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {kind}: {truncate(text, 20)}\n"
    with open(API_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def log_tool(name, args, result_preview, rc=None):
    """Записать вызов TOOL в журнал использования инструментов (``tool_usage.log``).

    Формат строки: ``[ISO-время] tool=<имя> args=<JSON> rc=<код возврата>
    result=<предпросмотр до 200 символов>``. Аргументы сериализуются с
    ``ensure_ascii=False``, чтобы русские символы оставались читаемыми (UTF-8).

    Args:
        name (str): имя инструмента (например, 'run_shell').
        args: аргументы вызова (dict или сырая строка JSON от модели).
        result_preview (str): результат/ошибка для предпросмотра в журнале.
        rc (int | str | None): код возврата команды либо метка 'timeout',
            'error', 'unknown'.
    """
    line = (
        f"[{datetime.now().isoformat(timespec='seconds')}] tool={name} "
        f"args={json.dumps(args, ensure_ascii=False)} rc={rc} "
        f"result={truncate(result_preview, 200)}\n"
    )
    with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def save_history(messages):
    """Сохранить историю чата в отдельный файл ``chat_history.jsonl``.

    Формат — JSON Lines: каждая строка = одно сообщение OpenAI-чата
    (role/content/tool_calls/tool_call_id), включая служебное системное
    сообщение. Файл перезаписывается целиком после каждого хода, поэтому
    всегда отражает актуальное состояние диалога. ``ensure_ascii=False``
    сохраняет кириллицу читаемой.

    Args:
        messages (list[dict]): текущая история сообщений.
    """
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


#: Реальный путь к shell-интерпретатору: bash, при его отсутствии — sh,
#: иначе гарантированный fallback /bin/sh. Определяется один раз при импорте;
#: тесты могут подменять константу через monkeypatch.setattr(agent, "SHELL_BIN", ...).
SHELL_BIN = shutil.which("bash") or shutil.which("sh") or "/bin/sh"


def run_shell(command, timeout=DEFAULT_SHELL_TIMEOUT):
    """Выполнить команду в полноценном shell — реализация инструмента run_shell.

    Вызывается из цикла main(), когда модель вернула ``tool_calls`` с именем
    'run_shell'. Команда запускается как ``<shell> -lc <command>``, поэтому
    доступны конвейеры, перенаправления, переменные, подстановки и т.д.

    Args:
        command (str): строка shell-команды от модели.
        timeout (int): максимальное время выполнения в секундах
            (по умолчанию DEFAULT_SHELL_TIMEOUT = 60).

    Returns:
        str: JSON-строка для отправки модели обратно как содержимое
        сообщения role="tool". Один из вариантов:
          * {"stdout": ..., "stderr": ..., "returncode": int} — успех;
          * {"error": "Таймаут выполнения (N c)"} — превышен таймаут;
          * {"error": "<текст исключения>"} — иная ошибка запуска.
        Каждый исход дополнительно фиксируется в tool_usage.log (:func:`log_tool`).
    """
    try:
        p = subprocess.run(
            [SHELL_BIN, "-lc", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = {
            "stdout": p.stdout,
            "stderr": p.stderr,
            "returncode": p.returncode,
        }
        log_tool("run_shell", {"command": command}, json.dumps(out, ensure_ascii=False), p.returncode)
        return json.dumps(out, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        err = json.dumps({"error": f"Таймаут выполнения ({timeout} c)"}, ensure_ascii=False)
        log_tool("run_shell", {"command": command}, err, "timeout")
        return err
    except Exception as e:
        err = json.dumps({"error": str(e)}, ensure_ascii=False)
        log_tool("run_shell", {"command": command}, err, "error")
        return err


def main():
    """Точка входа: интерактивный цикл диалога пользователя с моделью.

    Жизненный цикл сессии:

    1. **Конфигурация** — :func:`load_config` читает ``../config.yaml``;
       при любой проблеме печатает понятное сообщение в stderr и завершает
       процесс с кодом 1 (без трейсбека).
    2. **Клиент API** — создаётся OpenAI-совместимый клиент (api_key, base_url).
    3. **Начало сессии** — формируются стартовое сообщение (system-промпт)
       и список инструментов ``tools = [RUN_SHELL_TOOL]``; описание TOOL
       (function_call) передаётся через API уже в первом запросе и далее
       в каждом обращении к модели.
    4. **Цикл ввода** — читается строка пользователя; распознанные команды:
       'exit'/'quit'/'выход' — выход, 'clear' — очистка истории, пустая
       строка — пропуск. Всё остальное отправляется модели.
    5. **Внутренний цикл ответов** — модель может вернуть ``tool_calls``:
       тогда вызывается run_shell, результат добавляется как сообщение
       role='tool', и запрос повторяется, пока модель не ответит текстом.
       Сетевые сбои (APIConnectionError/APITimeoutError) обрабатываются
       без падения: неотправленное сообщение изымается из истории.
    6. **Персистентность** — после каждого хода история пишется в
       chat_history.jsonl; реплики — в api.log (обрезка до 20 символов);
       вызовы инструментов — в tool_usage.log.

    TODO(model-switch): Выбор модели во время сессии. План реализации:
        * переименовать локальную ``model`` в изменяемую ``current_model``;
        * добавить в цикл распознавания команд ветку::

              if user_input.startswith("/model "):
                  new_model = user_input.split(None, 1)[1].strip()
                  ... см. TODO(model-validate)/TODO(model-log)/TODO(model-history)
                  current_model = new_model
                  continue

          и использовать ``model=current_model`` в chat.completions.create();
        * печатать текущую модель в приветствии и после переключения.

    TODO(model-list): Команда '/models' -> client.models.list() и вывод имён;
        обработать отсутствие поддержки эндпоинта (исключение/пустой ответ).

    TODO(model-config-fallback): Опциональный cfg["allowed_models"] —
        ограничить множество допустимых имён моделей для '/model'.
    """
    # --- Шаг 1. Конфигурация ------------------------------------------------
    try:
        cfg = load_config(CONFIG_PATH)
    except ConfigError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # любая неожиданная ошибка (в т.ч. битый YAML) — без трейсбека
        print(
            f"Не удалось загрузить конфигурацию {CONFIG_PATH}: {e}\n"
            "Создайте/исправьте файл по примеру:\n" + EXAMPLE_CONFIG,
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Шаг 2. Клиент OpenAI-совместимого API ------------------------------
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    # Текущая модель сессии. TODO(model-switch): именно эта переменная будет
    # меняться командой '/model <имя>' без перезапуска программы.
    model = cfg.get("model", DEFAULT_MODEL)

    # --- Шаг 3. Начальное состояние диалога ---------------------------------
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # В начале сессии через API передаётся описание TOOL — function_call.
    # Список tools отправляется в каждом запросе chat.completions.create().
    tools = [RUN_SHELL_TOOL]
    print(f"Модель: {model}")
    print(f"URL:    {cfg['base_url']}")
    print("Инструмент run_shell подключён. Вход: 'exit' — выход, 'clear' — очистить историю.")
    # TODO(model-list): здесь же вывести подсказку '/models' при поддержке сервером.

    # --- Шаг 4. Внешний цикл: ввод пользователя -----------------------------
    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D / Ctrl-C там, где ожидается ввод — корректный выход.
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "выход"):
            break
        if user_input.lower() == "clear":
            # Полная очистка истории: оставляем только системный промпт.
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_history(messages)
            print("История очищена.")
            continue
        # TODO(model-switch): обработка '/model <имя>' и '/models' — до того,
        # как ввод уйдёт модели (см. docstring функции).

        # --- Отправка сообщения пользователя -------------------------------
        messages.append({"role": "user", "content": user_input})
        log_api("user", user_input)  # в api.log попадёт не более 20 символов

        # --- Шаг 5. Внутренний цикл ответов модели (tool-раунды) ------------
        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except (APIConnectionError, APITimeoutError) as e:
                # Сбой сети/API — не роняем программу с трейсбеком.
                cause = e.__cause__
                detail = f" ({cause})" if isinstance(cause, httpx.UnsupportedProtocol) else ""
                print(f"\nОшибка подключения к API{detail}: {e}. "
                      "Проверьте 'base_url' в config.yaml и доступность сервера. Попробуйте ещё раз.",
                      file=sys.stderr)
                messages.pop()  # убираем неотправленное сообщение пользователя
                save_history(messages)
                break
            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
                # Модель запросила выполнение инструментов. Сначала в историю
                # добавляется сообщение ассистента с tool_calls (в том виде,
                # в котором его ожидает API), затем — по одному сообщению
                # role='tool' на каждый вызов.
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in msg.tool_calls
                        ],
                    }
                )
                for tc in msg.tool_calls:
                    if tc.function.name == "run_shell":
                        # Разбор аргументов: модель присылает JSON-строку;
                        # некорректный JSON трактуем как пустой набор опций.
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        cmd = args.get("command", "")
                        timeout = int(args.get("timeout", DEFAULT_SHELL_TIMEOUT))
                        print(f"\n[TOOL run_shell] {cmd}")
                        result = run_shell(cmd, timeout)  # пишет tool_usage.log
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            }
                        )
                    else:
                        # Защитный путь: модель вызвала инструмент, которого
                        # нет в схеме. Возвращаем ошибку как результат
                        # инструмента, чтобы цикл диалога продолжился.
                        unknown = json.dumps({"error": f"Неизвестный инструмент: {tc.function.name}"}, ensure_ascii=False)
                        log_tool(tc.function.name, tc.function.arguments, unknown, "unknown")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": unknown,
                            }
                        )
                continue  # отправить результаты инструментов модели

            # Финальный текстовый ответ модели в этом ходе диалога.
            answer = msg.content or ""
            print(f"\nАссистент: {answer}")
            log_api("model", answer)  # в api.log попадёт не более 20 символов
            messages.append({"role": "assistant", "content": answer})
            break

        # --- Шаг 6. Персистентность после каждого хода -----------------------
        save_history(messages)

    save_history(messages)
    print(f"История сохранена в {HISTORY_PATH}")


if __name__ == "__main__":
    main()
