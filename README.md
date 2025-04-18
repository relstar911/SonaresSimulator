# SONARES - Sonic Resonance Evaluation System

SONARES ist ein wissenschaftliches Simulationstool zur Analyse akustischer Resonanz von Materialien in einer kontrollierten virtuellen Umgebung.

## Projektbeschreibung

SONARES ermöglicht die umfassende Charakterisierung von Materialien durch fortschrittliche Datenverarbeitung und experimentelle Modellierung. Das System simuliert, wie verschiedene Materialien auf Schallwellen reagieren, und visualisiert diese Reaktionen in einer interaktiven 3D-Umgebung.

## Hauptfunktionen

- **Materialanalyse**: Simulation der akustischen Eigenschaften verschiedener Materialien
- **Visualisierung**: 2D- und 3D-Visualisierung von Resonanzfeldern
- **Experimentelle Datenverwaltung**: Vergleich von Simulationsergebnissen mit experimentellen Daten
- **Datenbankintegration**: Speicherung und Abruf von Simulationsergebnissen und experimentellen Daten
- **Batch-Testing**: Automatisierte Tests für mehrere Materialien und Konfigurationen

## Technische Details

- Streamlit-basierte interaktive Benutzeroberfläche
- Wissenschaftliche Dateninterpolation mit SciPy
- Simulation mehrerer Materialien
- Datenbankunterstützung mit PostgreSQL
- Fortgeschrittene Wellenphysik-Simulationen

## Installation und Ausführung

1. Repository klonen:
   ```
   git clone https://github.com/relstar911/SonaresSimulator.git
   cd SonaresSimulator
   ```

2. Abhängigkeiten installieren:
   ```
   pip install -r dependencies.txt
   ```
   
   Hinweis: Die Datei `dependencies.txt` enthält alle erforderlichen Pakete für dieses Projekt.

3. Anwendung starten:
   ```
   streamlit run app.py
   ```

## Projektstruktur

- `app.py`: Hauptanwendungsdatei mit der Streamlit-Benutzeroberfläche
- `batch_app.py`: Enthält die Funktionalität für Batch-Tests
- `models/`: Enthält alle Kernmodule für die Simulation
  - `acoustic_simulation.py`: Kernmodul für die akustische Simulation
  - `database.py`: Datenbankfunktionalität
  - `material_database.py`: Verwaltung der Materialeigenschaften
  - `source_configurations.py`: Konfiguration akustischer Quellen
- `utils/`: Hilfsfunktionen und -module
- `experimental_data/`: Enthält experimentelle Datensätze für verschiedene Materialien

## Datenbank-Konfiguration

SONARES verwendet PostgreSQL für die Datenspeicherung. Die Datenbank-Verbindungsparameter werden über Umgebungsvariablen konfiguriert:

```
DATABASE_URL=postgresql://username:password@localhost:5432/sonares_db
```

Wenn keine Datenbankverbindung verfügbar ist, verwendet die Anwendung einen Fallback-Modus mit lokaler Speicherung.

## Lizenz

Copyright (c) 2025. Alle Rechte vorbehalten.