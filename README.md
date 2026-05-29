🏗️ Telegram AI Assistant for Regulatory Documents (RAG + Hybrid Search)

📋 Описание проекта

Telegram-бот для интеллектуального поиска по нормативной документации (СП, технические регламенты, строительные нормы) с использованием LLM и RAG-подхода.

Сервис отвечает на вопросы пользователей, извлекая релевантные фрагменты из базы знаний и добавляя их в контекст LLM. Для повышения качества retrieval реализован гибридный поиск: полнотекстовый (lexical) + семантический (vector search) с последующим ранжированием.

Ключевая бизнес-ценность:
-	сокращение времени поиска нормативных требований; 
-	повышение точности ответов LLM за счёт retrieval-контура; 
-	подготовка архитектуры для B2B knowledge base и AI-search сервисов. 
________________________________________
🎯 Основные задачи проекта

1.	Разработать Telegram-интерфейс для AI-сервиса с поддержкой диалогов, истории сообщений и управления сессиями. 
2.	Интегрировать LLM через OpenAI-compatible API с обработкой timeout, retry и конфигурацией моделей через .env. 
3.	Реализовать retrieval-контур для поиска релевантных фрагментов нормативной документации через Supabase и Qdrant. 
4.	Построить гибридный поиск: 
-	lexical retrieval; 
-	vector retrieval; 
-	объединение результатов через Reciprocal Rank Fusion (RRF). 
5.	Добавить observability и debugging AI/RAG-контура: 
-	логирование retrieval; 
-	тайминги; 
-	диагностику качества поиска; 
-	smoke-проверки интеграций. 
________________________________________
🏗️ Архитектура решения

Гибридный retrieval
```text
User Question
      ↓
Telegram Bot
      ↓
LLM Prompt Builder
      ↓
Retrieval Layer
 ├── Supabase (lexical search)
 └── Qdrant (vector search)
      ↓
Hybrid Ranking (RRF / Merge)
      ↓
LLM (OpenRouter / OpenAI API)
      ↓
Telegram Response
```
Хранилища

Supabase (PostgreSQL): 
-	chunks; 
-	chat_sessions; 
-	messages; 
-	metadata и retrieval-данные.
	
Qdrant: 
-	embeddings нормативных чанков; 
-	semantic vector search.
	
SQLite (fallback/local mode): 
-	локальная история диалогов; 
-	interaction logging. 
________________________________________
🛠️ Технологический стек

Python, pyTelegramBotAPI, OpenAI-compatible API / OpenRouter, Supabase (PostgreSQL), Qdrant, 
sentence-transformers, transformers, torch, SQLite, Python logging, JSONL  
________________________________________
🧪 Реализованный функционал

🤖 Telegram Bot & Dialogue Management

Разработали Telegram-бота с управлением диалогами: обработка сообщений, хранение истории, создание новых сессий и форматирование ответов LLM.
- Стек: Python, pyTelegramBotAPI, SQLite.
- Ценность: опыт разработки conversational AI-интерфейса и управления состоянием диалога.
________________________________________
🧠 LLM Integration Layer

Интегрировали OpenRouter / OpenAI-compatible API через собственный compatibility-layer с поддержкой timeout, retry и конфигурируемых моделей.
- Стек: Python, openai SDK, .env configuration.
- Ценность: работа с production-like AI API, обработкой transient failures и устойчивостью внешних интеграций.
________________________________________
🔎 Lexical Retrieval with Supabase

Реализовали retrieval по базе знаний через Supabase: поиск релевантных чанков по токенам и stop-словам с последующей передачей контекста в LLM.
- Стек: Supabase, PostgreSQL, Python, lexical search.
- Ценность: практический опыт построения retrieval-контура для повышения точности генерации.
________________________________________
🔢 Semantic Vector Search with Qdrant

Подключили векторный поиск по embeddings: формировали embedding пользовательского запроса и искали semantic nearest chunks в Qdrant.
- Стек: Qdrant, sentence-transformers, transformers, torch, ai-forever/ru-en-RoSBERTa.
- Ценность: опыт работы с vector databases, embeddings и semantic retrieval.
________________________________________
⚖️ Hybrid Retrieval & Reciprocal Rank Fusion

Улучшили качество retrieval через гибридное ранжирование: объединили lexical- и vector-поиск с использованием Reciprocal Rank Fusion и fallback merge-режима без дубликатов.
- Стек: Python, Supabase, Qdrant, RRF.
- Ценность: понимание retrieval quality, ranking strategies и компромиссов между полнотой поиска и размером LLM-контекста.
________________________________________
💬 Chat History & Session Storage

Спроектировали хранение истории диалогов: локально через SQLite и опционально через Supabase (chat_sessions, messages). 
- Стек: SQLite, Supabase, PostgreSQL.
- Ценность: опыт проектирования состояния AI-сервиса и резервных сценариев хранения данных.
________________________________________
📊 Observability & Retrieval Logging

Добавили logging и observability retrieval-контура: логирование найденных чанков и RRF-score. 
- Стек: Python logging, JSONL, SQLite.
- Ценность: диагностика retrieval quality и анализ поведения AI-системы.
________________________________________
🧪 Smoke Tests & Integration Checks

Создали smoke-проверки интеграций Supabase/Qdrant и сценарии тестирования записи/чтения сообщений.
- Стек: Python CLI, qdrant-client, supabase.
- Ценность: опыт диагностики внешних сервисов и проверки устойчивости интеграций.
________________________________________
🚀 Быстрый старт

Требования
-	Python 3.10+ 
-	Telegram Bot Token 
-	OpenRouter / OpenAI API Key 
Запуск
python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

cp env.example .env

python bot.py
________________________________________
🔧 Минимальная настройка .env

-	BOT_TOKEN=
-	OPENROUTER_API_KEY=
-	OPENROUTER_MODEL=

Для включения retrieval дополнительно настраиваются:

-	SUPABASE_URL=
-	SUPABASE_SERVICE_ROLE_KEY=
-	RAG_USE_QDRANT=1
-	QDRANT_URL=
-	QDRANT_COLLECTION=
________________________________________
📂 Основные файлы проекта

-	bot.py                     # Telegram runtime и orchestration
-	rag.py                     # Retrieval logic и hybrid search
-	qdrant_rag.py              # Vector retrieval
-	query_embedding.py         # Embedding generation
-	chat_storage.py            # SQLite chat storage
-	telegram_supabase_chat.py  # Supabase chat storage
-	supabase_helper.py         # Supabase client factory
-	interaction_db.py          # Retrieval logging
-	verify_rag_connections.py  # Smoke tests
________________________________________
🏷️ Теги

#rag #llm #telegrambot #supabase #qdrant #python #vector-search #hybrid-search #retrieval #openai #transformers #nlp #sqlite #postgresql #observability #ai-assistant

