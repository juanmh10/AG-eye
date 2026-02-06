# poe-test AG monitor

MVP: monitora a vida do Animate Guardian (AG) e envia `Esc` quando atingir o limiar.

## Requisitos (Windows)
- Python 3.11+
- Path of Exile em modo janela

## Clonar
```bash
git clone https://github.com/juanmh10/AG-eye.git
cd poe-test
```

## Instalação
```bash
pip install -r requirements.txt
```

Ou via venv (Windows):
```bash
python -m venv venv
venv\Scripts\pip.exe install -r requirements.txt
```

## Config inicial
Copie o arquivo de exemplo e ajuste conforme necessario:
```bash
copy config\\config.example.json config\\config.json
```

## Ajustar ROI (primeira vez)
O app abre uma janela para escolher o threshold e, opcionalmente, selecionar o ROI.
Na selecao de ROI, clique na borda esquerda e depois na borda direita da barra do AG.
O ROI salvo fica em `config/config.json`.
Marque "Calibrar" quando estiver com vida cheia para ajustar a escala.
Esse modo tambem calibra a cor real da barra (nao depende de vermelho fixo).

## Rodar
```bash
run.bat
```

## Debug
```bash
run_debug.bat
```

## Configuração
Edite `config/config.json` para:
- `threshold_percent`: limiar de vida
- `trigger_frames`: frames consecutivos abaixo do limiar
- `poll_interval_ms`: intervalo de captura
- `use_grounding`: ligar/desligar grounding via ícone
