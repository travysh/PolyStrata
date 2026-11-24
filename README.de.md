<h1 align="center">PolyStrata</h1>

[English](README.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · **Deutsch** · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md) · [Türkçe](README.tr.md)

<p align="center"><b>Große KI-Modelle auf deinem eigenen PC</b><br>
Qwen3.8-Flash-Next, Hy4-Flash, GLM 5.3 Flash, DeepSeek V4 Flash, LFM2.5, Flint, Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder, MiniCPM-V und Kolibri 1 · kostenlos und Open Source</p>

<p align="center"><a href="https://github.com/simplexai-labs/linger"><img height="28" alt="Used by Linger" src="docs/media/used-by-linger.svg"></a></p>

<p align="center"><img src="docs/media/speed.svg" width="820" alt="Tokens pro Sekunde beim Schreiben einer Antwort: Qwen3.6-35B-A3B schreibt 106 auf einem Laptop mit 8-GB-Karte und 536 auf einer RTX 5090"></p>

<p align="center"><img src="docs/media/claude-code-demo.gif" width="820" alt="Claude Code im Terminal neben dem API Monitor von PolyStrata"><br>
<sub>Claude Code mit Qwen3.6-35B-A3B UD-Q3_K_M auf dem Laptop mit der 8-GB-Karte, in Echtzeit: Es fügt einem kleinen Programm die Option <code>--json</code> hinzu, schreibt einen Test dafür und führt die Tests aus. Der Server hatte den Systemprompt von Claude Code beim Start gelesen (<code>"warm_start": true</code>).</sub></p>

<p align="center"><img src="docs/media/linger-inkling-3d.gif" width="820" alt="Linger auf PolyStrata schreibt ein 3D-Dorf"><br>
<sub><a href="https://github.com/simplexai-labs/linger">Linger</a> auf PolyStrata: Inkling 2 Mini UD-Q3_K_XL schreibt auf dem Laptop mit 8-GB-Karte, mit etwa 33 Tokens pro Sekunde, ein Low-Poly-Dorf als eine three.js-Datei, und Linger öffnet sie.</sub></p>

PolyStrata lässt große KI-Modelle auf einem normalen PC laufen, Modelle, die sonst meist einen Server brauchen.
Sie chatten, schreiben Code und arbeiten mit deinen Apps und Coding-Agents zusammen. Nichts verlässt deinen PC.

Es lässt vierzehn Modelle laufen, auf zwei Engines:

- **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** auf einer NVIDIA- oder AMD-Grafikkarte
  ab 12 GB, unter Windows oder Linux. Es versteht auch Bilder. Seine Engine ist die von
  [Strata](https://github.com/Niko1221/Strata), unverändert.

Die anderen dreizehn laufen auf einer Engine, die für PolyStrata geschrieben wurde, mit einer NVIDIA-Karte oder [zwei](docs/MULTI_GPU.md#the-strata-poly-models): alle dreizehn
unter Linux, alle außer GLM 5.3 Flash und DeepSeek V4 Flash auch unter Windows. Sie sind experimentell: Sie sind bisher auf einem einzigen PC mit
einer 32-GB-Karte gelaufen, neun der kleineren außerdem auf einem Laptop mit einer 8-GB-Karte und Groot auf einem PC mit
einer RTX 5060 mit 8 GB.

- **[GLM 5.3 Flash](docs/models/glm.de.md)**, das größte: eine Datei von 79 bis 114 GB, die zum größten Teil im RAM bleibt.
- **[DeepSeek V4 Flash](docs/models/deepseek.de.md)**, eine Datei von 91 GB, die ebenfalls zum größten Teil im RAM bleibt. Sein
  Gedächtnis für eine Unterhaltung ist klein: Ein Kontext von 128K braucht nur 0,7 GB mehr VRAM als 32K.
- **[LFM2.5-24B-A2B](docs/models/lfm25.de.md)** von Liquid AI, eine Datei von 11 bis 25 GB. Die meisten seiner Schichten haben
  keine Attention, darum braucht eine lange Unterhaltung wenig VRAM: Ein Kontext von 128K braucht nur 2 GB mehr als
  32K. Seine Lizenz erlaubt die kommerzielle Nutzung nur Firmen mit weniger als 10 Millionen Dollar Jahresumsatz.
- **[Flint 40B-A2.5B](docs/models/flint.de.md)** von Recursive, gemacht für Coding-Agenten: eine Datei von 18 bis 42 GB. Die
  meisten seiner Schichten schauen nur 513 Tokens zurück, darum braucht eine lange Unterhaltung wenig VRAM: Ein
  Kontext von 128K braucht nur 2,6 GB mehr als 32K.
- **[Qwen3.6-35B-A3B](docs/models/qwen36.de.md)**, eine Datei von 13 bis 23 GB. Es versteht auch Bilder.
- **[Groot 19B-A3B 261023](docs/models/groot.md)**, experimentell im Installer, mit drei Quantisierungen von Bartowski,
  Q4_K_M, Q6_K und Q8_0 (12,5 bis 20,2 GB), mit MTP und optionalem Bildprojektor. Q4_K_M lief mit Text auf einer
  RTX 5060 mit 8 GB und unter Linux auf einer RTX 5090; Bilder, Q6_K und Q8_0 stehen noch aus.
- **[Inkling 2 Mini](docs/models/inkling.md)** von Thinking Machines, experimentell: eine Datei von 9 bis 18 GB. Es lief
  unter Linux mit einer 32-GB-Karte von NVIDIA und unter Windows mit einer 8-GB-Laptopkarte. Es liest auch Bilder, eine Anfrage nach der anderen.
- **[Clover-35B-A3B-Preview](docs/models/clover.de.md)** von DiffuSpace, Qwen3.6-35B-A3B weiter trainiert, damit es
  auch einen Block von Tokens auf einmal schreibt: eine Datei von 13 bis 21 GB. Es liest nur Text.
- **[Ornith 1.5](docs/models/ornith.de.md)**, mit der Architektur von Qwen3.6-35B-A3B und anderen Gewichten: eine Datei von
  14 bis 24 GB. Es versteht auch Bilder.
- **[Gemma 4 26B-A4B](docs/models/gemma4.de.md)**, eine Datei von 11 bis 17 GB. Es versteht auch Bilder.
- **[Tiel Coder 35B-A3B](docs/models/tiel.de.md)**, Ornith 1.5, von peculiar-ragdoll für Coding-Agents quantisiert:
  eine Datei von 14 bis 23 GB. Es versteht auch Bilder.
- **[MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.de.md)** von OpenBMB, mit der Architektur von Qwen3.6-35B-A3B und eigenen
  Gewichten, gemacht, um Bilder zu lesen: eine Datei von 16 bis 29 GB.
- **[Kolibri 1](docs/models/kolibri.de.md)** von Aleph Alpha, das Deutsch so gut schreibt wie Englisch: eine Datei von 29 bis
  48 GB, die neben einer kleinen Karte größtenteils im RAM bleibt.
- **[Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18)** von Tencent, 220B-A12B, auf der eigenen Engine des Projekts Strata (`strata-hy4`): eine Datei von
  130 bis 181 GB, ab 64 GB RAM mit einer 8-GB-Karte. Es liest auch Bilder.

Der Server kann im laufenden Betrieb zwischen installierten Modellen wechseln
([wie](docs/BACKENDS.md#changing-the-model), englisch).

PolyStrata wurde von [VecSzn](https://github.com/VecSzn) erstellt.
PolyStrata baut auf [Strata](https://github.com/Niko1221/Strata) von Niko1221 auf.
Warum das ein eigenes Repo ist, AMD- und kleinere Karten, Fine-Tunes und andere Fragen: [FAQ](docs/FAQ.md) (englisch).

## Wie schnell ist es?

Ein Token ist etwa ¾ eines Wortes.

- **Schreibt Antworten:** wie schnell die Antwort in einem kurzen Chat erscheint. 60 Tokens pro Sekunde sind
  schneller, als du lesen kannst.
- **Liest deinen Prompt:** wie schnell es aufnimmt, was du schickst (hier ein Dokument, Code oder Chatverlauf mit
  32K Tokens).

Die Tabelle zeigt für jedes Modell die Größe, die der Installer nimmt, in Tokens pro Sekunde. Die Zahlen jeder Größe, der Vergleich mit llama.cpp und wie gemessen wurde, stehen auf der Seite des Modells.

| Modell | Laptop mit 8-GB-Karte<br>schreibt / liest | RTX 5090<br>schreibt / liest | Alle Größen |
| --- | ---: | ---: | --- |
| LFM2.5-24B-A2B | 86 / 4,035 | 492 / 25,275 | [LFM2.5](docs/models/lfm25.de.md) |
| Flint 40B-A2.5B | 64 / 2,643 | 261 / 15,033 | [Flint](docs/models/flint.de.md) |
| Qwen3.6-35B-A3B | 106 / 2,483 | 536 / 15,319 | [Qwen3.6](docs/models/qwen36.de.md) |
| Clover-35B-A3B-Preview | 83 / 2,242 | 308 / 13,918 | [Clover](docs/models/clover.de.md) |
| Ornith 1.5 | 87 / 2,362 | 366 / 14,438 | [Ornith](docs/models/ornith.de.md) |
| Gemma 4 26B-A4B | 92 / 2,187 | 370 / 11,478 | [Gemma 4](docs/models/gemma4.de.md) |
| Tiel Coder 35B-A3B | 89 / 2,517 | 440 / 15,109 | [Tiel Coder](docs/models/tiel.de.md) |
| MiniCPM-V 4.7 35B-A3B | 79 / 2,263 | 331 / 14,615 | [MiniCPM-V](docs/models/minicpmv.de.md) |
| Kolibri 1 | 66 / 851 | 207 / 6,133 | [Kolibri 1](docs/models/kolibri.de.md) |
| GLM 5.3 Flash | läuft nicht: braucht eine 32-GB-Karte | 52 / 863 | [GLM](docs/models/glm.de.md) |
| DeepSeek V4 Flash | läuft nicht: braucht eine 32-GB-Karte | 54 / 1,926 | [DeepSeek](docs/models/deepseek.de.md) |
| Inkling 2 Mini | 53 / 892 | 196 / 6,706 | [Inkling 2 Mini](docs/models/inkling.md) |
| Groot 19B-A3B 261023 | 97 / 2,785 | 458 / 15,634 | [Groot](docs/models/groot.md) |
| Hy4-Flash | läuft nicht: braucht 64 GB RAM | 90-111 / 1,694 | [Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18) |

Laptop: RTX 4070 Laptop (8 GB), Core i7-14700HX, 32 GB RAM. Workstation: RTX 5090 (32 GB), Xeon Platinum 8470Q (25 vCPU), 90 bis 120 GB RAM. Flint ist auf dem Laptop UD-Q3_K_XL, die Größe, die der Installer bei 32 GB RAM nimmt, und auf der Workstation UD-Q4_K_XL. Kolibri 1 ist auf beiden Q2_K; der Installer der Workstation nimmt Q4_K_M, das nicht gemessen wurde.

Qwen3.8-Flash-Next und Hy4-Flash laufen auf den Engines des Projekts [Strata](https://github.com/Niko1221/Strata) und wurden von diesem Projekt gemessen. Qwen3.8-Flash-Next auf einem PC mit RTX 5070 (12 GB) und 64 GB RAM: IQ2_XS schreibt mit 79 und liest einen 32K-Prompt mit 2,090 ([alle Größen](docs/models/qwen38.de.md)). Hy4-Flash auf einer RTX 5090: schreibt mit 90 bis 111 und liest einen 128K-Prompt mit 1,694 ([Release](https://github.com/Niko1221/Strata/releases/tag/v0.2.18)).

Die elf Modelle der eigenen Engine von PolyStrata wurden mit ihrem Benchmark gemessen, den du auch auf deinem PC
laufen lassen kannst: `python polystrata.py benchmark --model <name> --prompt-tokens 24000`
([was er schickt und ausgibt](docs/BACKENDS.md#benchmark), englisch).
Was andere auf ihren eigenen Karten gemessen haben und wie du auf deinem PC mit llama.cpp vergleichst:
[Ergebnisse aus der Community](docs/COMMUNITY_BENCHMARKS.md#the-models-polystrata-adds) (englisch).
Auf dem Laptop las PolyStrata Prompts 2.7- bis 3.5-mal so schnell wie llama.cpp b11540 und schrieb 1.5- bis
1.8-mal so schnell, mit Qwen3.6 und mit Tiel Coder ([die beiden Läufe](docs/COMMUNITY_BENCHMARKS.md#comparing-with-llamacpp), englisch).

## Wie viel RAM es belegt

Seit 0.1.45-poly.7 belegt die Engine der dreizehn Modelle weniger RAM: Sie hält die Teile der Modelldatei, die schon auf
der Grafikkarte liegen, nicht mehr im Speicher. Auf dem Laptop belegte Qwen3.6-35B-A3B nach einem Prompt von 32,000
Tokens mit 0.1.45-poly.6 17.9 GiB, jetzt 13.1 GiB, und es schreibt genauso schnell. Unter Linux ging es von 17.8 auf
12.6 GiB.

Um weniger zu belegen, füge `"--ram-experts-mib", "6144"` zur `"args"`-Liste von `strata-<model>.json` hinzu
([wo](docs/INSTALL.md#where-things-are-stored)) und starte neu. Die Engine hält dann so viele MiB RAM für die Experten
außerhalb des VRAM, auch während sie antwortet, und sagt das beim Start. Die übrigen bleiben im Dateicache des
Systems, wo andere Programme den Speicher nutzen können, und werden von dort gelesen, wenn ein Token sie braucht.
Passen die Experten in den freien RAM, ist das standardmäßig aus, weil es langsamer schreibt; wir arbeiten weiter
daran. Passen sie nicht hinein, schaltet die Engine es selbst ein (siehe unten). Auf dem Laptop (RTX 4070 Laptop
8 GB, 32 GB RAM, Windows 11), Qwen3.6-35B-A3B UD-Q3_K_M:

| | Standard | `--ram-experts-mib 6144` |
| --- | ---: | ---: |
| RAM der Engine nach dem Laden | 11.7 GiB | **6.5 GiB** |
| höchstens beim Schreiben: kurzer Chat / nach einem 32K-Prompt | 14.3 / 14.0 GiB | **6.7 / 7.3 GiB** |
| zwischen Anfragen | 12.8 GiB | **7.1 GiB** |
| Schreibt, Token/s: der erste kurze Chat | 71.6 | 69.4 (3 % langsamer) |
| nach dem 32K-Prompt | 65.4 | 63.7 (3 % langsamer) |
| ein kurzer Chat danach | 84.2 | 72.9 (13 % langsamer) |
| eine lange Antwort (1,024 Token eines kurzen Chats) | 90.6 | 78.3 (14 % langsamer) |

Bei dieser Messung belegten andere Programme einen Teil des VRAM, darum schreiben beide Spalten langsamer als mit
freier Karte. In zwei anderen Läufen schrieb es nach dem 32K-Prompt 6 bis 8 % langsamer und im Chat danach 8 bis
10 %. Unter Linux (eine RTX PRO 6000, auf etwa 8 GB begrenzt) belegte es 7.0 bis 7.7 GiB statt 12.3 bis 14.3 und
schrieb 7 bis 14 % langsamer. Mit festgehaltenem Experten-Cache, wie ein Release geprüft wird, sind die Antworten
Token für Token gleich.

Passen die Experten außerhalb des VRAM nicht in den freien RAM, hält die Engine jetzt die meistgenutzten in einem
Budget aus diesem RAM minus 4 GiB und liest die übrigen von der SSD, wenn ein Token sie braucht; das sagt sie beim
Start. Vorher bildete sie alle ab, und das System las sie Seitenfehler für Seitenfehler zurück. Das Speicherlimit
eines Containers zählt mit. Gemessen unter Linux auf einer RTX PRO 6000, auf etwa 8 GB begrenzt, mit der Engine auf
12 GB und auf 10 GB RAM begrenzt (was ihr auf einem PC mit 16 GB bleibt) und der Modelldatei bei jedem Start
außerhalb des Dateicaches: Qwen3.6-35B-A3B UD-Q3_K_M, Token/s im ersten kurzen Chat, nach einem 24,000-Token-Prompt
und in einem kurzen Chat danach.

| RAM für die Engine | alle Experten abgebildet (vorher) | das Budget, das sie jetzt nimmt |
| --- | ---: | ---: |
| 12 GB | 51 / 75-78 / 94-98 | **57-60 / 77-85 / 131-139** |
| 10 GB | 39 / 48 / 69 | **51 / 64 / 92** |

Ohne Limit schreibt dieselbe Maschine 63 / 169 / 203. Unter Windows, wo derselbe Code läuft, wurde es nicht
gemessen. `"--ram-experts-mib", "-1"` bildet wie bisher alle Experten ab.

Umgekehrt hält `"--lock-experts"` die Experten außerhalb des VRAM im RAM, sodass das System ihren Speicher nicht an
andere Programme gibt und die Engine sie später nicht wieder von der SSD liest. Auf einer einzelnen NVIDIA-Karte
kopiert die Engine sie beim Laden in gepinnten Speicher, den das System nicht auslagert und den `ulimit -l` nicht
begrenzt; ein Experte, der später das VRAM verlässt, wird zwischen zwei Anfragen aus der Modelldatei nachkopiert. Wo
sie diese Kopie nicht anlegt (AMD-Karte, Mac, Schichten auf mehrere Karten verteilt, zu wenig freier RAM), sperrt sie
stattdessen die Seiten der Modelldatei; unter Linux endet das bei `ulimit -l`, und die Startzeile der Engine nennt
das Limit und wie man es anhebt.

Zum Lesen von Prompts braucht die Engine außerdem 350 bis 400 MiB weniger VRAM: Der Expertencache nimmt sie (Qwen3.6
hält 2,712 statt 2,444 Experten im VRAM), oder `--vram-reserve-mib` lässt sie anderen Programmen. Wie es funktioniert
und gemessen wurde: [docs/BACKENDS.md](docs/BACKENDS.md#less-of-the-pcs-ram-and-vram).

## Was du brauchst

| Modell | Grafikkarte | RAM (Größe des Installers) | Festplatte | System |
| --- | --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/models/qwen38.de.md) | NVIDIA RTX 20 bis 50, oder AMD RX 6800 / 6900, RX 7700 XT / 7800 XT / 7900 XT / XTX, RX 9060 XT / 9070 / 9070 XT, Radeon AI PRO R9700; ab 12 GB | ab 32 GB; mit 64 GB laufen alle Größen | etwa 80 GB | Windows, Linux |
| [GLM 5.3 Flash](docs/models/glm.de.md) | NVIDIA RTX 20 oder neuer; nur auf einer 32-GB-Karte gelaufen | 96 GB neben einer 32-GB-Karte | 79 bis 114 GB, auf einer SSD | Linux |
| [DeepSeek V4 Flash](docs/models/deepseek.de.md) | wie oben | 72 GB neben einer 32-GB-Karte | 91 GB, auf einer SSD | Linux |
| [LFM2.5-24B-A2B](docs/models/lfm25.de.md) | NVIDIA RTX 20 oder neuer, ab 8 GB | etwa 20 GB neben einer 8-GB-Karte | 11 bis 25 GB | Windows, Linux, Mac (Beta) |
| [Flint 40B-A2.5B](docs/models/flint.de.md) | wie oben | etwa 22 GB neben einer 8-GB-Karte | 18 bis 42 GB | Windows, Linux |
| [Qwen3.6-35B-A3B](docs/models/qwen36.de.md) | wie oben; AMD als Beta | etwa 21 GB neben einer 8-GB-Karte | 13 bis 23 GB | Windows, Linux, Mac (Beta) |
| [Clover-35B-A3B-Preview](docs/models/clover.de.md) | NVIDIA RTX 20 oder neuer, 8 GB oder mehr | etwa 20 GB neben einer 8-GB-Karte | 13 bis 21 GB | Windows, Linux |
| [Ornith 1.5](docs/models/ornith.de.md) | wie oben; AMD als Beta | etwa 22 GB neben einer 8-GB-Karte | 14 bis 24 GB | Windows, Linux |
| [Gemma 4 26B-A4B](docs/models/gemma4.de.md) | wie oben; AMD als Beta | etwa 18 GB neben einer 8-GB-Karte | 11 bis 17 GB | Windows, Linux |
| [Tiel Coder 35B-A3B](docs/models/tiel.de.md) | wie oben; AMD als Beta | etwa 21 GB neben einer 8-GB-Karte | 14 bis 23 GB | Windows, Linux |
| [MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.de.md) | wie oben; AMD als Beta | etwa 25 GB neben einer 8-GB-Karte | 16 bis 29 GB | Windows, Linux |
| [Kolibri 1](docs/models/kolibri.de.md) | NVIDIA RTX 20 oder neuer, ab 8 GB | etwa 33 GB neben einer 8-GB-Karte | 29 bis 48 GB | Windows, Linux |
| [Groot 19B-A3B 261023](docs/models/groot.md) (experimentell) | gemeinsame CUDA/HIP-Backends; Q4_K_M mit Text auf einer RTX 5060 mit 8 GB gelaufen (gemessene Spitze 7,05 GiB VRAM, 21,8 GiB RAM) | etwa 17 GB neben einer 8-GB-Karte für Q4_K_M, Schätzung des Installers | 12,5 bis 20,2 GB; Bilder 0,9 GB mehr | Windows, Linux; Bilder, Q6_K und Q8_0 ungetestet |
| [Inkling 2 Mini](docs/models/inkling.md) (experimentell) | NVIDIA ab 8 GB; gelaufen auf einer RTX 5090 und einer RTX 4070 Laptop (8 GB, UD-Q3_K_XL) | etwa 21 GB neben einer 8-GB-Karte für UD-Q4_K_XL, Schätzung des Installers | 9 bis 18 GB | Windows, Linux, Mac (Beta) |
| [Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18) (Engine des Projekts Strata) | NVIDIA V100 bis RTX 50, ab 8 GB; AMD RX 6000 bis 9000 und Strix Halo; Intel Arc | ab 64 GB für SQ-64, 96 GB für SQ-96, 128 GB für SQ-128 | 130 bis 181 GB, auf einer SSD | Windows, Linux, Mac |

Der RAM ist die Schätzung des Installers; eine größere Karte braucht weniger. Alles andere richtet der Installer ein: Unter Windows lädt er die Engine fertig gebaut herunter und kompiliert sie nur auf einem Prozessor ohne AVX2, für die eigenen Modelle von PolyStrata auf einer AMD-Karte oder wenn die Engine deiner Kopie neuer als das letzte Release ist; fehlende Werkzeuge bietet er an. Zwei oder drei Karten können sich ein Modell teilen ([Multi-GPU](docs/MULTI_GPU.md)). Ältere Grafikkarten, Intel Arc, Apple-Silicon-Macs (Qwen3.8-Flash-Next, als Beta auch Qwen3.6, LFM2.5 und Inkling 2 Mini), Strix Halo und ältere Prozessoren ohne AVX2: [die ganze Liste](docs/INSTALL.md#what-you-need). Karte, RAM und Größen jedes Modells im Detail stehen auf seiner Seite.

Was Qwen3.6, Clover, Ornith 1.5, Gemma 4, Tiel Coder und MiniCPM-V jeweils brauchen und wie es geprüft wurde: [docs/BACKENDS.md](docs/BACKENDS.md#install) (englisch).

## Installieren

### Lass deine KI es einrichten

Nutzt du einen KI-Coding-Assistenten (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Füge dort diesen Text ein:

```text
Set up PolyStrata on this PC for me: https://github.com/simplexai-labs/PolyStrata - follow docs/AI_SETUP.md in that repository.
```

Er prüft deine Grafikkarte, deinen RAM und deine Festplatte und wählt das passende Modell. Dann installiert und
startet er es und sagt dir, wie du deine Apps verbindest. KI-Tools können PolyStrata auch über seinen
[MCP-Server](docs/MCP_SERVER.md) installieren, starten und stoppen.

### Oder mach es selbst

[Lade PolyStrata herunter](https://github.com/simplexai-labs/PolyStrata/archive/refs/heads/main.zip) und entpacke es (oder nutze
`git clone`). **Windows:** Doppelklick auf **`START-HERE.bat`**. **Linux:** Führe im PolyStrata-Ordner **`./setup.sh`** aus.

**Mac (Apple Silicon, M1 bis M5, macOS 15 oder neuer):** Installiere einmal Apples Command Line Tools (`xcode-select --install` im Terminal; Xcode selbst brauchst du nicht). Öffne dann das Terminal im PolyStrata-Ordner und führe **`./setup.sh`** aus. Es baut die Mac-Engine auf deinem Mac (etwa 7 Minuten), stellt dieselben Fragen, lädt das Modell und startet es. Beim nächsten Mal startet `./setup.sh` sofort. Welche Größe zu deinem Mac passt und wie schnell es ist: [docs/MAC.md](docs/MAC.md). Standardmäßig richtet es Qwen3.8-Flash-Next ein. `./setup.sh --family qwen36` oder `--family lfm25` richtet stattdessen Qwen3.6 oder LFM2.5 auf der Metal-Engine von strata-poly ein, als Beta: Auf einem M5 Ultra schrieb Qwen3.6 UD-Q2_K_XL in einem kurzen Chat 134 Tokens/s und las einen Prompt mit 3,768 Tokens mit 2,978 Tokens/s. `--family inkling` richtet Inkling 2 Mini genauso ein: Auf dem M5 Ultra schrieb UD-Q3_K_XL in einem kurzen Chat 74 Tokens/s, las einen Prompt mit 32,787 Tokens mit 2,440 Tokens/s und schrieb danach 61 Tokens/s; Bilder liest es auf der CPU. Setup bietet auf dem Mac auch GLM, DeepSeek, Ornith, Gemma 4, Tiel Coder und MiniCPM-V an, aber die liefen dort noch nicht mit ihren echten Gewichten. Kolibri, Flint und Clover brauchen eine NVIDIA- oder AMD-Karte.

Die Schritte sind für NVIDIA und AMD gleich. Der Installer erkennt deine Karte und richtet die passende Engine
dafür ein. Er stellt dir ein paar Fragen:

- welches Modell und welche Größe,
- wie viel Kontext (wie viel Text sich das Modell merkt),
- ob es Bilder lesen soll (GLM 5.3 Flash, DeepSeek V4 Flash, LFM2.5, Flint, Clover und Kolibri 1 lesen keine).

Drück jedes Mal Enter, um die empfohlene Antwort zu nehmen. Dann lädt er das Modell herunter (etwa 70 GB für
Qwen3.8-Flash-Next, 79 bis 114 GB für GLM 5.3 Flash, 91 GB für DeepSeek V4 Flash, 29 bis 48 GB für Kolibri 1, 11 bis 42 GB für die anderen acht) und
startet es. Bricht der Download ab, starte ihn einfach neu: Er macht dort weiter, wo er aufgehört hat. Sobald das Modell
bereit ist, ist seine API unter `http://127.0.0.1:8080/v1` erreichbar. Wenn du eine App zum Chatten willst, empfehlen
wir [Linger](https://github.com/simplexai-labs/linger). Jede App, die die API von OpenAI oder Anthropic spricht, geht auch.

Alle vierzehn stehen in der Modellliste des Installers. `./setup.sh --family glm`, `deepseek`, `lfm25`, `flint`, `qwen36`, `clover`, `ornith`,
`gemma4`, `tiel`, `minicpmv`, `kolibri`, `groot` oder `inkling` führt direkt zu einem davon ([was er fragt und schreibt](docs/GLM.md#install), englisch).

> **Während das Modell startet, kann dein PC 1-3 Minuten langsam sein oder nicht reagieren** (beim ersten Mal am
> längsten). PolyStrata lädt 35-55 GB in deinen RAM und reserviert einen Teil davon für die Grafikkarte. Das ist normal.
> Warte und schließ das Fenster nicht. Im Fenster siehst du, was PolyStrata gerade tut.

**Beim nächsten Mal** startest du wieder `START-HERE.bat` (oder `./setup.sh`). Es startet sofort und lädt nichts
doppelt herunter. Schließ das Fenster, um das Modell zu stoppen. `UPDATE.bat` (`./setup.sh --update`) aktualisiert
PolyStrata, ohne es zu starten.
Kommst du von 0.1.45-poly.10 oder älter: Die Dateien von setup liegen jetzt in `terminal/` ([was du ändern musst](docs/INSTALL.md#coming-from-0145-poly10-or-older)).
Updates, mehrere Karten, wo die Dateien landen und alle Optionen:
[docs/INSTALL.md](docs/INSTALL.md).

**Läuft es auf deinem PC schnell?** Starte `tools/BENCH.bat` (`tools/bench.sh`): etwa 5-10 Minuten; es misst Chat, 4K/16K-Prompts und eine Sitzung im Agenten-Stil mit deinen eigenen Einstellungen und schreibt einen Bericht zum Teilen ([wie](docs/COMMUNITY_BENCHMARKS.md#run-the-benchmark)).

## Welches Modell soll ich nehmen?

Dasselbe Modell gibt es in mehreren Größen, mehr oder weniger komprimiert: Kleinere sind schneller und brauchen weniger RAM, größere sind näher am Original. Der Installer nimmt eine passende für deinen RAM und deine Karte, wenn du Enter drückst; für eine andere Größe gib bei der Installation `--model <Größe>` an.

| Modell | Der Installer nimmt | Mit weniger RAM | Nur mit Namen |
| --- | --- | --- | --- |
| [Qwen3.8-Flash-Next](docs/models/qwen38.de.md) | 32 GB RAM: Coder; 48 bis 64 GB: IQ2_XS; ab 96 GB: IQ3_S |  | Q2_0, IQ3_XXS, Unsloths 4- bis 8-Bit-Dateien |
| [GLM 5.3 Flash](docs/models/glm.de.md) | 3.0BIT | UD-IQ2_XXS, dann REAP50-Q3_K_M |  |
| [DeepSeek V4 Flash](docs/models/deepseek.de.md) | UD-IQ2_XXS (die einzige Größe) |  |  |
| [LFM2.5-24B-A2B](docs/models/lfm25.de.md) | UD-Q4_K_XL | UD-Q3_K_XL | QAD-Q4_0, UD-Q6_K_XL, Q8_0 |
| [Flint 40B-A2.5B](docs/models/flint.de.md) | UD-Q4_K_XL ab etwa 40 GB RAM | UD-Q3_K_XL | UD-Q6_K_XL, Q8_0 |
| [Qwen3.6-35B-A3B](docs/models/qwen36.de.md) | UD-Q3_K_M | UD-Q2_K_XL | UD-Q4_K_M |
| [Clover-35B-A3B-Preview](docs/models/clover.de.md) | Q3_K_M | APEX-I-NANOPLUS | Q4_K_M |
| [Ornith 1.5](docs/models/ornith.de.md) | APEX-I-COMPACT | APEX-I-MINI | APEX-I-QUALITY |
| [Gemma 4 26B-A4B](docs/models/gemma4.de.md) | UD-Q3_K_M | UD-Q2_K_XL | UD-Q4_K_M, QAT-UD-Q4_K_XL |
| [Tiel Coder 35B-A3B](docs/models/tiel.de.md) | UD-Q3_K_XL | UD-IQ3_XXS | UD-Q4_K_XL |
| [MiniCPM-V 4.7 35B-A3B](docs/models/minicpmv.de.md) | Q4_K_M | mix_q3gud_im | Q6_K |
| [Kolibri 1](docs/models/kolibri.de.md) | Q4_K_M ab etwa 48 GB RAM | Q2_K | Q3_K_M |
| [Groot 19B-A3B 261023](docs/models/groot.md) | Q4_K_M | — | Q6_K, Q8_0 |
| [Inkling 2 Mini](docs/models/inkling.md) | UD-Q4_K_XL | UD-Q3_K_XL | UD-Q2_K_XL |
| [Hy4-Flash](https://github.com/Niko1221/Strata/releases/tag/v0.2.18) | SQ-64 ab 64 GB RAM, SQ-96 ab 96 GB, SQ-128 ab 128 GB |  |  |

Was jede Größe ist, ihr Download, der RAM neben anderen Karten und wie nah sie an die größte kommt: die Seite jedes Modells, verlinkt in der Tabelle unter „Wie schnell ist es?“.

## So benutzt du es

Alle vierzehn Modelle benutzt du gleich, über dieselbe API.

<p align="center"><a href="https://github.com/simplexai-labs/linger"><img src="docs/media/linger-desktop.jpg" width="900" alt="Linger auf einem Mac: ein Chatraum, in dem Agents von ihrer Arbeit berichten, daneben die Chrome-Fenster, in denen sie arbeiten"></a><br>
<sub><a href="https://github.com/simplexai-labs/linger">Linger</a>: Mehrere Agents teilen sich einen Computer und den Browser, in dem du schon angemeldet bist. Ein Bild aus dem Linger-Projekt.</sub></p>

- **Als persönlicher Agent:** Wir empfehlen [Linger](https://github.com/simplexai-labs/linger). Lade es für Mac (Apple Silicon), Windows oder Linux aus
  [dem neuesten Release](https://github.com/simplexai-labs/linger/releases/latest) herunter.
- **Im Terminal:** `python polystrata.py --model <name>` im PolyStrata-Ordner chattet mit einem installierten Modell,
  `--list-models` listet sie auf ([mehr](docs/INSTALL.md#the-terminal)).
- **Im Browser:** `http://127.0.0.1:8080` hat einen Live-**Monitor** für das Modell und
  deine GPU/CPU/RAM, **Requests**, das den [API-Anfragemonitor](docs/DETAILS.md#api-request-monitor) in einem eigenen Tab
  öffnet, sobald er eingeschaltet ist, und **About** mit den Einstellungen und Adressen. Mit mehr als einem installierten
  Modell wechselt die Liste oben zwischen ihnen.
- **Deine Apps und Coding-Agents:** Füge einen „OpenAI-kompatiblen“ Anbieter mit der Basis-URL
  **`http://127.0.0.1:8080/v1`** hinzu. Jeder API-Key und jeder Modellname funktioniert.
  - Apps, die die API von Anthropic nutzen: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`, dazu `"anthropic_thinking": "on_request"` in `strata-<model>.json`
    ([warum](docs/FAQ.md#can-i-use-it-with-claude-code-codex-or-other-apps))).
  - Codex CLI und andere Apps, die die OpenAI Responses API nutzen: `/v1/responses`
    ([Einrichtung](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Nachdenken:** Wähle **aus, niedrig, mittel oder hoch** unter „Reasoning Effort“ in deiner
  App. Aus ist am schnellsten. Hoch ist am besten für schwierige Fragen.
- **Vom Handy oder einem anderen PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Setz immer einen
  Key.

Wo sie sich unterscheiden:

| | Qwen3.8-Flash-Next | GLM 5.3 Flash | DeepSeek V4 Flash | LFM2.5-24B-A2B | Flint 40B-A2.5B | Kolibri 1 | Qwen3.6-35B-A3B, Clover-35B-A3B-Preview, Ornith 1.5, Gemma 4 26B-A4B, Tiel Coder 35B-A3B, MiniCPM-V 4.7 35B-A3B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Bilder** | Antworte im Setup mit Ja auf „Images?“. Häng dann Bilder in deiner App an. Mit weniger als 24 GB Grafikspeicher läuft der Bild-Encoder auf dem Prozessor, bei NVIDIA wie bei AMD. | Noch nicht. | Es liest nur Text. | Es liest nur Text. | Es liest nur Text. | Es liest nur Text. | Antworte im Setup mit Ja auf „Images?“, dann wie bei Qwen3.8-Flash-Next. Auf einer Karte mit weniger als 16 GB läuft der Bild-Encoder auf dem Prozessor, und ein Bild dauert ein paar Sekunden. Clover liest nur Text. |
| **Eine Anfrage nach der anderen** | Standardmäßig beantwortet PolyStrata eine Anfrage, die anderen warten. Um mehrere gleichzeitig zu beantworten, setz `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). Auf einer 12-GB-Karte wird dadurch jede Antwort langsamer. | Eine nach der anderen, die anderen warten. | Eine nach der anderen, die anderen warten. | Eine nach der anderen, die anderen warten. | Eine nach der anderen, die anderen warten. | Eine nach der anderen, die anderen warten. | Eine nach der anderen, die anderen warten. |
| **Mehrere Unterhaltungen** | Standardmäßig bleiben mehrere im RAM, soweit Platz ist; Folgenachrichten und Agenten, die sich abwechseln, starten also in Sekunden ([Details](docs/DETAILS.md#conversation-cache)). | Die letzte bleibt erhalten. Das Setup fragt, ob mehrere im RAM bleiben sollen, damit ein Agent und seine Subagenten sich abwechseln, ohne dass ihre Prompts neu gelesen werden ([wie](docs/GLM.md#several-conversations-at-once)). | Wie GLM 5.3 Flash. | Wie GLM 5.3 Flash. | Wie GLM 5.3 Flash. | Wie GLM 5.3 Flash. | Wie GLM 5.3 Flash. |
| **Lange Prompts** | PolyStrata liest die erste Nachricht eines Chats komplett, etwa 1 Minute pro 30,000 Tokens. Folgenachrichten starten in Sekunden. | Die erste Nachricht wird komplett gelesen, etwa 40 Sekunden pro 30,000 Tokens. | Die erste Nachricht wird komplett gelesen, etwa 29 Sekunden pro 30,000 Tokens. | Die erste Nachricht wird komplett gelesen: 30,000 Tokens brauchen 1,3 bis 1,5 Sekunden auf der Workstation und 8 bis 10 Sekunden auf dem 8-GB-Laptop. | Die erste Nachricht wird komplett gelesen: 30,000 Tokens brauchen etwa 2,3 Sekunden auf der Workstation und 15 bis 22 Sekunden auf dem 8-GB-Laptop. | Die erste Nachricht wird komplett gelesen: 30,000 Tokens brauchen mit Q2_K auf der Workstation etwa 6 Sekunden und auf dem 8-GB-Laptop etwa 40 Sekunden. | Die erste Nachricht wird ganz gelesen: 30,000 Tokens dauern auf der Workstation 4 bis 5 Sekunden und auf dem 8-GB-Laptop 14 bis 27 Sekunden. |

Mehr: [wo was gespeichert wird](docs/INSTALL.md#where-things-are-stored), [die API](docs/DETAILS.md#using-it).

### Optional: mehr Tempo mit einer Grafikkarte, wenn das ganze Modell in den RAM passt

Zwei Opt-in-Einstellungen (von RubenTorney, #1904 und #1905). Sie helfen, wenn du eine Grafikkarte nutzt und dein RAM alle Experten hält (der Arena-Modus von PolyStrata, zum Beispiel Q2_0 oder IQ3_XXS auf einem PC mit 64 GB). Öffne deine Konfiguration `strata-<model>.json` im Ordner `terminal` von PolyStrata ([wo](docs/INSTALL.md#where-things-are-stored)) und füge hinzu:

- **Schnellere Ausgabe:** `"--adapt-async", "1"` in die Liste `"args"`. Der Experten-Cache wird dann in einem Hilfs-Thread aktualisiert, statt die Antwort anzuhalten. Gemessen auf einer RTX 5070 (Windows, Q2_0): +4-5 % Ausgabe, +7-10 % bei Code und langen Dokumenten. Antworten können leicht vom Standard abweichen (sie bleiben von Lauf zu Lauf gleich).
- **Schnelleres Prompt-Lesen unter Windows:** `"STRATA_PIN_BY_NEED": "1"` im Block `"env"`. Windows kann nur etwa den halben RAM für die Grafikkarte sperren; damit werden zuerst die Experten gesperrt, die die Karte wirklich braucht. Gemessen auf der RTX 5070: +4,4 % Prompt-Lesen.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Starte PolyStrata danach neu (`START-HERE.bat` oder `./setup.sh`). Zum Rückgängigmachen entferne die beiden Einträge.

## Etwas ist schiefgelaufen?

- **Mein PC ist beim ersten Start von PolyStrata eingefroren.** Das ist normal, während das Modell lädt. Warte und
  schließ das Fenster nicht. Nach 10 Minuten immer noch eingefroren? Starte den PC neu, schließ andere Programme und
  versuch es noch einmal, oder nimm eine kleinere Größe.
- **Es hat beim Herunterladen oder Installieren aufgehört.** Starte `START-HERE.bat` (oder `./setup.sh`) noch
  einmal. Es macht dort weiter, wo es aufgehört hat.
- **Es ist sehr langsam und die Festplattenlampe blinkt ständig, oder es meldet „the engine stopped
  unexpectedly“.** Dein PC hat nicht genug freien RAM. Schließ andere Programme (Browser brauchen viel) oder nimm
  eine kleinere Größe (Q2_0 oder IQ2_XS).
- **Es meldet, dass Port 8080 schon belegt ist.** PolyStrata läuft bereits. Such nach seinem Fenster.

Weitere Probleme und ihre Lösungen: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Kommst du nicht weiter?
Öffne ein [Issue](https://github.com/simplexai-labs/PolyStrata/issues) und hänge `strata-<model>.log` aus dem Ordner `terminal`
an. Du hast ein Sicherheitsproblem gefunden? Melde es vertraulich: [SECURITY.md](SECURITY.md).

## Wie funktioniert es?

Modelle wie dieses laufen normalerweise auf Servern mit Hunderten Gigabyte Grafikspeicher. Deine Grafikkarte hat
8-32 GB. PolyStrata bringt das Modell trotzdem unter, indem es **die Arbeit auf deinen ganzen PC verteilt**. Stell dir
eine Küche vor: Was du ständig brauchst, bleibt auf der Arbeitsfläche, der Rest wartet in der Vorratskammer.

Bild und Erklärung für jedes Modell: [docs/HOW_IT_WORKS_MODELS.de.md](docs/HOW_IT_WORKS_MODELS.de.md).

## Auf PolyStrata gebaut

<p align="center"><a href="https://github.com/simplexai-labs/linger"><img src="https://raw.githubusercontent.com/simplexai-labs/linger/main/assets/linger-story.webp" width="820" alt="Linger in 28 Sekunden: Er richtet ein lokales Modell ein, verbindet deine Apps, macht aus Mails und Meetings Erinnerungen und fragt, bevor er eine E-Mail sendet."></a><br>
<sub>Linger in 28 Sekunden: Er richtet ein lokales Modell ein, verbindet deine Apps, macht aus Mails und Meetings Erinnerungen und fragt, bevor er eine E-Mail sendet.</sub></p>

**[Linger](https://github.com/simplexai-labs/linger)** ist ein persönlicher Open-Source-Agent, der ganz auf deinem PC läuft; seine lokalen
Modelle laufen auf PolyStrata. Er findet in deinen Apps, was du brauchst, merkt sich, was wichtig ist, und erledigt
Dinge, während du weg bist. [Doku](https://docs.heylinger.com/) · [GitHub](https://github.com/simplexai-labs/linger)

Du hast etwas auf PolyStrata gebaut? Öffne ein [Issue](https://github.com/simplexai-labs/PolyStrata/issues), dann nehmen wir es hier auf.

## Danksagung und Lizenz

Qwen3.8-Flash-Next ist vom [Qwen-Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Komprimiert wurde es von
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5) und Unsloth.
GLM 5.3 Flash läuft aus den Dateien von [neuralll](https://huggingface.co/neuralll/GLM-5.3-Flash-GSQ-RCO-3.0bit-Q4Kattn-GGUF),
Unsloth und patrickbdevaney. Qwen3.6-35B-A3B ist vom Qwen-Team, Gemma 4 von Google und DeepSeek V4 Flash von DeepSeek, alle drei in der
Quantisierung von [Unsloth](https://huggingface.co/unsloth). LFM2.5-24B-A2B ist von
[Liquid AI](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), in der Quantisierung von Unsloth und in Liquid AIs
eigenem QAD-Q4_0. Flint 40B-A2.5B ist von
[Recursive](https://huggingface.co/RecursiveSI/flint), in der Quantisierung von Unsloth. Inkling 2 Mini ist von [Thinking Machines](https://huggingface.co/thinkingmachines/Inkling-2-Mini), in der Quantisierung von Unsloth. Groot 19B-A3B 261023 ist von GrootTeam, in der Quantisierung von [bartowski](https://huggingface.co/bartowski/GrootTeam_Groot-19B-A3B-261023-GGUF). Kolibri 1 ist von [Aleph Alpha](https://huggingface.co/Aleph-Alpha/Kolibri-1), in der GGUF-Umwandlung von [Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF). Clover-35B-A3B-Preview ist von [DiffuSpace](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview), in der Quantisierung von [maternion](https://huggingface.co/maternion/Clover-35B-A3B-Preview-GGUF) und IsValorums APEX. Ornith 1.5 ist von ornith-ai, in der APEX-Quantisierung von
[mudler](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF). Tiel Coder 35B-A3B ist Ornith 1.5 in der Quantisierung von
[peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP);
[u/Sensitive_Song4219](https://www.reddit.com/r/LocalLLM/comments/1x22inc/comment/pf3r111/) hat herausgefunden, dass es auf PolyStrata läuft. MiniCPM-V 4.7 35B-A3B ist von [OpenBMB](https://huggingface.co/openbmb/MiniCPM-V-4.7-35B-A3B-gguf),
in ihrer eigenen Quantisierung; [iMilnb](https://github.com/iMilnb) hat es zu PolyStrata gebracht. PolyStrata baut auf [Strata](https://github.com/Niko1221/Strata) von Niko1221 und seinen Mitwirkenden
auf und nutzt Teile von [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). Alle Danksagungen:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). PolyStrata ist Open Source unter der [MIT-Lizenz](LICENSE). Ein
paar Teile und alle Modelle haben eigene Lizenzen ([welche](docs/HOW_IT_WORKS.md#license)).
