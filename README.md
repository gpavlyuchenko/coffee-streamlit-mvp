
# Coffee Landed Cost — Streamlit MVP

MVP на **Streamlit**: бесплатные котировки арабики/робусты (Stooq), дифференциал, контейнеры 20’/40’, пресеты юрисдикций (ЕАЭС/ОАЭ),
маршруты для CFR/CIF, локальные сборы, расчёт CV→пошлина→НДС→итог, экспорт CSV/Excel/PDF.

## Запуск локально
```bash
pip install -r requirements.txt
streamlit run app.py
# откройте ссылку из терминала (http://localhost:8501)
```

## Деплой на Streamlit Community Cloud
1) Залейте проект в GitHub (репозиторий, ветка main).
2) Перейдите на https://share.streamlit.io → New app → укажите `user/repo`, ветку и `app.py` → Deploy.

## Примечание
- Данные Stooq — с задержкой, только для ознакомительных расчётов.
- Инкотермс применены упрощённо для MVP (FOB/CFR/CIF).
