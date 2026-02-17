# IkePicFall

**IkePicFall** -- Conversor de Imagem para Audio via Espectrograma

Baseado na ideia original do **PicFall** de **PY4ZBZ**, implementado em Python por **Henrique B. Gravina - PU3IKE** utilizando **Claude AI** (Anthropic).

## Descricao

IkePicFall permite transformar uma imagem qualquer (mas sem muitos detalhes!) em um arquivo de audio no formato WAV (mono, 16 bits, 11025 am/s) que, ao ser reproduzido, permite visualizar esta imagem em um espectrograma deslizante vertical (waterfall), ou em um espectrograma estatico como os famosos "40 / 20m Spectrometer" do Edson PU1JTE. Para estes, usar a opcao "Inverted" em "waterfall vertical time scale".

O sinal de audio gerado pelo IkePicFall pode ser transladado para RF por meio de um transmissor USB, sem alterar seu conteudo espectral.

### Como funciona

A imagem resultante tera o equivalente de no maximo 200x230 pixels, e e gerada a partir da imagem original de qualquer tamanho, aberta com "Open", ou com "Paste" caso exista uma na area de transferencia.

Cada pixel gera uma portadora cuja frequencia e proporcional a sua posicao horizontal. O numero de portadoras e igual a largura espectral dividida por Delta F. A largura espectral pode ser ajustada entre 1000 e 2800 Hz. O espacamento Delta F entre portadoras pode ser escolhido entre 14 e 20 Hz. O numero de portadoras e de pixels horizontais varia portanto entre 50 e 200. Cada portadora e modulada em AM pela luminancia dos pixels da coluna vertical correspondente. As transicoes entre pixels verticais adjacentes sao feitas com envoltoria na forma de "coseno levantado" para minimizar o espalhamento espectral. Quanto maior o numero de portadoras, mais detalhes podem ser vistos, mas em compensacao, menor sera a potencia disponivel para cada uma!

### Parametros ajustaveis

| Parametro | Faixa | Padrao |
|---|---|---|
| Largura Espectral | 1000 -- 2800 Hz | 2000 Hz |
| Delta F | 14 -- 20 Hz | 16 Hz |
| Frequencia Central | 1300 -- 1700 Hz | 1500 Hz |
| Duracao | 1 -- 60 s | 10 s |
| Brilho | -100 a +100 | 0 |
| Contraste | -100 a +100 | 0 |

A janela "Spectrum" mostra o espectro acumulado do sinal, linha a linha, para facilitar os ajustes de brilho e contraste da imagem. Abaixo dela, sao mostradas as frequencias minima, central e maxima do sinal resultante. A janela abaixo do espectro mostra aproximadamente como a imagem seria reproduzida em um espectrograma ideal.

O valor de Delta F depende muito da resolucao do espectrograma onde a imagem sera visualizada, e deve ser determinado experimentalmente.

**Atencao:** Apesar de ser usada uma tecnica especial para diminuir a relacao pico/eficaz do sinal complexo resultante, esta ainda e evidentemente alta. Portanto, e de suma importancia ajustar corretamente a potencia no caso de ser usado um transmissor USB. Por exemplo, um transmissor de 100W nao podera gerar mais de 20W medios!

## Instalacao

### Requisitos

- Python 3.8 ou superior
- Sistema operacional: Linux, Windows ou macOS

### Dependencias

```bash
pip install numpy pillow scipy
```

No Linux, pode ser necessario instalar o Tkinter separadamente:

```bash
# Debian/Ubuntu
sudo apt install python3-tk

# Fedora
sudo dnf install python3-tkinter

# Arch
sudo pacman -S tk
```

### Download

Clone o repositorio ou copie o arquivo `ikepicfall.py` para uma pasta de sua preferencia.

## Uso

### Executar

```bash
python3 ikepicfall.py
```

### Passo a passo

1. **Abrir imagem**: Use `File > Open` (Ctrl+O) para carregar uma imagem (PNG, JPG, BMP, etc.) ou `File > Paste` (Ctrl+V) para colar da area de transferencia.
2. **Ajustar parametros**: Use os controles deslizantes para ajustar largura espectral, Delta F, frequencia central, duracao, brilho e contraste.
3. **Inverter/Espelhar**: Use `View > Invert` para inverter as cores e `View > Flip Vertical` para espelhar a imagem verticalmente (util para adequar ao tipo de espectrograma).
4. **Gerar audio**: Clique no botao **Generate**.
5. **Salvar WAV**: Use `File > Save WAV` (Ctrl+S) para salvar o arquivo de audio.
6. **Visualizar**: Abra o arquivo WAV no Audacity ou outro software com visualizacao de espectrograma para ver a imagem.

### Dicas

- Use imagens com alto contraste e poucos detalhes para melhores resultados.
- Ajuste a duracao de acordo com a escala vertical do espectrograma onde sera visualizado.
- No Audacity: selecione a trilha, clique na seta ao lado do nome e escolha "Spectrogram" para visualizar.
- Para espectrogramas estaticos tipo "waterfall vertical", use a opcao "Flip Vertical" conforme necessario.

## Creditos

- Ideia original: **PY4ZBZ** (PicFall)
- Implementacao em Python: **Henrique B. Gravina - PU3IKE**
- Assistencia de programacao: **Claude AI** (Anthropic)
