# Sistema de Média Móvel para Estabilização da Classe Predominante

## Visão Geral

Foi implementado um sistema de média móvel das últimas 10 frames para estabilizar a exibição da classe predominante, reduzindo oscilações e falsos positivos/negativos na detecção.

## Funcionamento

### 1. Histórico de Classes por Frame

O sistema mantém um histórico das últimas 10 frames com as seguintes informações:
```python
frame_data = {
    'smudge': stats.get('smudge', 0),
    'simbolos': stats.get('simbolos', 0), 
    'blackdot': stats.get('blackdot', 0),
    'frame': self.frame_count
}
```

### 2. Cálculo da Média Móvel

Para cada classe, é calculada a média móvel das últimas 10 frames:
```python
smudge_avg = sum(f['smudge'] for f in self.class_history) / len(self.class_history)
simbolos_avg = sum(f['simbolos'] for f in self.class_history) / len(self.class_history)
blackdot_avg = sum(f['blackdot'] for f in self.class_history) / len(self.class_history)
```

### 3. Determinação da Classe Predominante

A classe predominante é determinada pela maior média móvel:
```python
class_scores = {
    'Smudge': smudge_avg,
    'Símbolos': simbolos_avg,
    'BlackDot': blackdot_avg
}
predominant_class = max(class_scores, key=class_scores.get)
```

### 4. Cálculo da Confiança

A confiança é calculada baseada em dois fatores:

#### a) Frequência de Detecção
- Conta quantos frames recentes (últimos 5) têm a classe predominante detectada
- `confidence = predominant_detections / len(recent_frames)`

#### b) Intensidade da Detecção
- Normaliza a média móvel para um fator de 0-1
- `intensity_factor = min(max_score / 2.0, 1.0)`

#### Confiança Final
- `final_confidence = confidence * intensity_factor`

### 5. Thresholds e Validações

- **Threshold mínimo**: Uma classe só é considerada predominante se `max_score >= 0.1`
- **Histórico mínimo**: Precisa de pelo menos 3 frames para calcular a média móvel
- **Janela de estabilização**: 10 frames para média móvel, 5 frames para confiança

## Exibição na Interface

### Label da Classe Predominante

A interface exibe a classe predominante estabilizada com:
- **Ícone**: 🎯 para destacar
- **Nome da classe**: Smudge, Símbolos, BlackDot ou Nenhuma
- **Confiança**: Percentual baseado na estabilidade da detecção
- **Cor**: Verde escuro (#2E7D32) quando detectada, cinza (#666666) quando nenhuma

### Exemplo de Exibição

```
🎯 Predominante: Smudge (85.2%)
```

## Vantagens do Sistema

### 1. Estabilidade
- Reduz oscilações rápidas entre classes
- Evita mudanças bruscas na exibição

### 2. Confiabilidade
- Baseado em múltiplos frames, não apenas um frame
- Considera tanto frequência quanto intensidade da detecção

### 3. Responsividade
- Janela de 10 frames mantém responsividade adequada
- Atualização em tempo real na interface

### 4. Transparência
- Mostra a confiança da detecção
- Permite ao usuário avaliar a qualidade da detecção

## Logs e Debug

O sistema registra logs informativos a cada 60 frames:
```
🎯 Classe Predominante: Smudge (confiança: 0.85)
```

## Configuração

Os parâmetros podem ser ajustados no código:
- `moving_average_window = 10`: Tamanho da janela de média móvel
- `threshold mínimo = 0.1`: Threshold para considerar classe predominante
- `recent_frames = 5`: Frames para cálculo de confiança

## Integração

O sistema está integrado ao pipeline de detecção existente:
1. **Detecção**: Classes são detectadas normalmente
2. **Histórico**: Estatísticas são adicionadas ao histórico
3. **Cálculo**: Média móvel é calculada
4. **Exibição**: Classe predominante é mostrada na interface
5. **Log**: Informações são registradas no log

Este sistema melhora significativamente a estabilidade e confiabilidade da exibição da classe predominante, proporcionando uma experiência mais consistente para o usuário.
