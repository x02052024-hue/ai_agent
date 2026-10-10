#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
agent.py — интерактивный чат-агент через OpenAI-совместимый API.

Настройки (ключ и URL) читаются из ../config.yaml, пример которого приведён
в начале этого файла в виде текстового блока (см. EXAMPLE_CONFIG ниже).

Возможности:
  * Интерактивный режим: ввод пользователя -> запрос к модели -> ответ.
  * Поддержка TOOL с полноценным shell: модель может вызвать функцию run_shell,
    команда выполняется локально, результат возвращается модели.
  * Лог API-взаимодействий (api.log): текст пользователя и модели ограничивается
    20 символами.
  * Системный промпт требует отсутствие нечитаемых символов при кодировке UTF-8
    (русский язык).
  * История чата сохраняется в отдельный файл chat_history.jsonl.
  * Журнал использования TOOL (tool_usage.log).
  * В начале сессии через API передаётся описание TOOL — function_call.

Пример ../config.yaml:

    api_key: "sk-my-secret-key"
    base_url: "http://localhost:8000/v1"
    model: "gpt-4o-mini"
"""

import json
import os
import subprocess
import sys
from datetime import datetime

import yaml
from openai import OpenAI

EXAMPLE_CONFIG = """\
# Пример ../config.yaml:
api_key: "sk-my-secret-key"
base_url: "http://localhost:8000/v1"
model: "gpt-4o-mini"
"""

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config.yaml")
API_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "api.log")
TOOL_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tool_usage.log")
HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_history.jsonl")

SYSTEM_PROMPT = (
    "Ты — полезный ассистент. Отвечай на русском языке. "
    "Во всех ответах недопустимы нечитаемые символы при кодировке UTF-8 "
    "(русский язык должен отображаться корректно, без кракозябр и битых байтов). "
    "Для выполнения системных команд используй инструмент run_shell."
)

RUN_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": (
            "Выполняет переданную строку команды в полноценном shell (/bin/bash) "
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


def truncate(text, limit=20):
    """Ограничение текста до `limit` символов для лога."""
    text = (text or "").replace("\n", " ")
    return text[:limit]


class ConfigError(Exception):
    """Ошибка конфигурации (отсутствует файл или обязательные параметры)."""


def load_config(path):
    if not os.path.exists(path):
        raise ConfigError(
            f"Не найден файл конфигурации: {path}\n"
            "Создайте его по примеру:\n" + EXAMPLE_CONFIG
        )
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ConfigError(f"Файл {path} не содержит корректных параметров.")
    for key in ("api_key", "base_url"):
        if key not in cfg:
            raise ConfigError(f"В {path} отсутствует обязательный параметр '{key}'.")
    return cfg


def log_api(kind, text):
    """Лог API-взаимодействия: пользователь и модель — не более 20 символов."""
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {kind}: {truncate(text, 20)}\n"
    with open(API_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def log_tool(name, args, result_preview, rc=None):
    """Журнал использования TOOL."""
    line = (
        f"[{datetime.now().isoformat(timespec='seconds')}] tool={name} "
        f"args={json.dumps(args, ensure_ascii=False)} rc={rc} "
        f"result={truncate(result_preview, 200)}\n"
    )
    with open(TOOL_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)


def save_history(messages):
    """Сохранение истории чата в отдельный файл."""
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def run_shell(command, timeout=60):
    """Полноценный shell: выполнение команды через /bin/bash."""
    try:
        p = subprocess.run(
            ["/bin/bash", "-lc", command],
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
    try:
        cfg = load_config(CONFIG_PATH)
    except ConfigError as e:
        print(e)
        sys.exit(1)
    client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    model = cfg.get("model", "gpt-4o-mini")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # В начале сессии через API передаётся описание TOOL — function_call.
    tools = [RUN_SHELL_TOOL]
    print(f"Модель: {model}")
    print(f"URL:    {cfg['base_url']}")
    print(f"Инструмент run_shell подключён. Вход: 'exit' — выход, 'clear' — очистить историю.")

    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "выход"):
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            save_history(messages)
            print("История очищена.")
            continue

        messages.append({"role": "user", "content": user_input})
        log_api("user", user_input)

        # Цикл обработки ответов модели (в т.ч. вызовов инструментов).
        while True:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            choice = response.choices[0]
            msg = choice.message

            if msg.tool_calls:
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
                        try:
                            args = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            args = {}
                        cmd = args.get("command", "")
                        timeout = int(args.get("timeout", 60))
                        print(f"\n[TOOL run_shell] {cmd}")
                        result = run_shell(cmd, timeout)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": result,
                            }
                        )
                    else:
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

            answer = msg.content or ""
            print(f"\nАссистент: {answer}")
            log_api("model", answer)
            messages.append({"role": "assistant", "content": answer})
            break

        save_history(messages)

    save_history(messages)
    print(f"История сохранена в {HISTORY_PATH}")


if __name__ == "__main__":
    main()
