# Chatterbox Português do Brasil

Worker de serverless da Runpod para o modelo [ResembleAI/Chatterbox-Multilingual-pt-br](https://huggingface.co/ResembleAI/Chatterbox-Multilingual-pt-br). A fala sai em português do Brasil. A clonagem usa o áudio enviado no pedido.

Os pesos não vão neste repositório. A imagem baixa o modelo brasileiro na build e só fica pronta se os pesos encaixarem.

## Pedido

`POST https://api.runpod.ai/v2/{endpoint_id}/runsync`

```json
{
  "input": {
    "text": "Oi, tudo bem?",
    "reference_audio_url": "https://exemplo/voz.wav",
    "reference_cache_key": "slot-1.wav@2026-09-26T15:11:33Z",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "temperature": 0.8
  }
}
```

`language_id` é sempre `pt`. Sem `reference_audio_url`, a voz padrão também é brasileira.

`reference_cache_key` é opcional. Com a mesma chave, o worker reaproveita a voz já preparada e não baixa a amostra de novo. Troque a chave quando a amostra mudar. O worker guarda as 8 vozes mais recentes.

A resposta traz `audio_base64` (WAV mono de 16 bits), `sample_rate`, `language` (`pt-BR`) e `cloned`.

O worker não sobe sem GPU CUDA e gera uma frase de aquecimento antes de aceitar pedidos.
